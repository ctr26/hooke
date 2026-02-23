import dataclasses
import json
import logging
import os
import re
import tempfile
import time
from typing import Literal

import numpy as np
import ornamentalist
import torch
from torch import nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm

from hooke_forge.model.tokenizer import DataFrameTokenizer
from hooke_forge.data.dataset import CellPaintConverter
from hooke_forge.utils.distributed import Distributed, rank_zero
from hooke_forge.utils.ema import KarrasEMA
from hooke_forge.utils.evaluation import (
    compute_cossim,
    compute_fd,
    compute_prdc,
)
from hooke_forge.utils.encoders import DINOv2Detector, Phenom2Detector, StabilityCPEncoder
from hooke_forge.utils.infinite_dataloader import infinite_dataloader
from hooke_forge.utils.profiler import get_profiler

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Module-level path for file-based metrics logging (used when wandb is not available)
_metrics_log_path: str | None = None


def set_metrics_log_path(path: str | None) -> None:
    """Set the path for file-based metrics logging (JSONL format).

    Call this before running evaluation/training to enable file-based logging
    when wandb is not initialized.
    """
    global _metrics_log_path
    _metrics_log_path = path
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _log.info(f"Metrics will be logged to {path}")


def _serialize_for_json(obj):
    """Convert non-JSON-serializable objects to serializable format."""
    if isinstance(obj, wandb.Image):
        return (
            f"<wandb.Image: {obj._path}>" if hasattr(obj, "_path") else "<wandb.Image>"
        )
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


@rank_zero()
def log(*, step: int, msg: str | None = None, data: dict | None = None):
    if msg is not None:
        _log.info(f"[{step=}] {msg}")
    if data is not None:
        if wandb.run is not None:
            wandb.log(data, step=step)
        elif _metrics_log_path is not None:
            # Log to JSONL file when wandb is not available
            serializable_data = {k: _serialize_for_json(v) for k, v in data.items()}
            record = {"step": step, **serializable_data}
            with open(_metrics_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")


@dataclasses.dataclass
class TrainState:
    ddp: DDP  # Distributed wrapper around the model
    ema: KarrasEMA  # EMA wrapper
    opt: torch.optim.Optimizer
    global_step: int
    tokenizer: DataFrameTokenizer  # Tokenizer for conditioning factors

    def _to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "ddp": self.ddp.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "tokenizer": self.tokenizer.state_dict(),
        }

    @rank_zero()
    def save_ckpt(self, path: str) -> None:
        dir = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dir, suffix=".tmp") as f:
            temp_path = f.name
            torch.save(self._to_dict(), temp_path)
        os.rename(temp_path, path)

    def load_ckpt(self, path: str, device: torch.device, strict: bool = True) -> None:
        """Load a checkpoint.

        Args:
            strict: If False, ignore mismatched keys (useful for loading a
                    Px-pretrained checkpoint into a Tx or joint model).
        """
        state = torch.load(path, weights_only=False, map_location=device)
        self.global_step = state["global_step"]
        self.ddp.load_state_dict(state["ddp"], strict=strict)
        self.ema.load_state_dict(state["ema"], strict=strict)
        if strict:
            self.opt.load_state_dict(state["opt"])
        else:
            _log.info("Skipping optimizer state load (strict=False)")
        if "tokenizer" in state:
            self.tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    def save_latest_ckpt(self, dir: str) -> None:
        path = os.path.join(dir, f"step_{self.global_step}.ckpt")
        self.save_ckpt(path)

    def load_latest_ckpt(self, dir: str, device: torch.device, strict: bool = True) -> None:
        pattern = r"step_(\d+).ckpt"
        fnames = [
            f.name
            for f in os.scandir(dir)
            if f.is_file() and re.fullmatch(pattern, f.name)
        ]
        if len(fnames) == 0:
            print(f"No existing checkpoints found in {dir}, skipping load.")
            return None

        latest = max(fnames, key=lambda x: int(re.fullmatch(pattern, x).group(1)))  # type: ignore
        path = os.path.join(dir, latest)
        print(f"Found previous checkpoint, loading latest from {path}")
        self.load_ckpt(path, device, strict=strict)


# ---------------------------------------------------------------------------
# Px evaluation helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_px(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    vae: StabilityCPEncoder,
    D: Distributed,
    use_ema: bool,
) -> None:
    model: nn.Module = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for batch in tqdm(
        loader,
        desc="Evaluating (Px)",
        total=len(loader),
        unit="step",
        disable=D.rank != 0,
    ):
        meta = batch["meta"]
        meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}
        x1 = batch["img"]
        x1 = x1.to(D.device, non_blocking=True)
        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)

        loss = model.loss("px", x0=x0, x1=x1, meta=meta)
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    prefix = "ema" if use_ema else "ddp"
    log(step=state.global_step, data={f"val/{prefix}_px_loss": val_loss})
    D.barrier()


@rank_zero()
@torch.inference_mode()
def visualise_phenomics(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    vae: StabilityCPEncoder,
    cp2rgb: CellPaintConverter,
    D: Distributed,
    use_ema: bool,
):
    model: nn.Module = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    batch = next(iter(loader))
    x1 = batch["img"]
    meta = batch["meta"]

    x1 = x1.to(D.device, non_blocking=True)
    meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

    x1 = vae.encode(x1)
    x0 = torch.randn_like(x1)
    preds, nfe = model.generate("px", x0=x0, meta=meta)
    preds = vae.decode(preds)
    preds = cp2rgb(preds)  # uint8 [B, 3, H, W]
    preds = preds.to(torch.float32) / 255  # save_image wants float32

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    save_path = os.path.join(output_dir, f"samples/{state.global_step}.png")
    save_image(preds, save_path)
    prefix = "ema" if use_ema else "ddp"
    log(
        step=state.global_step,
        data={f"val/{prefix}_samples": wandb.Image(save_path), "val/nfe": nfe},
    )
    return


@torch.inference_mode()
def compute_phenomics_metrics(
    *,
    state: TrainState,
    name: str,
    loader: torch.utils.data.DataLoader,
    vae: StabilityCPEncoder,
    cp2rgb: CellPaintConverter,
    D: Distributed,
    use_ema: bool,
) -> None:
    model: nn.Module = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    dino = DINOv2Detector(device=D.device)
    phenom = Phenom2Detector(device=D.device)

    real_features_dino = []
    pred_features_dino = []
    real_features_phenom = []
    pred_features_phenom = []

    N = len(loader.dataset)  # type: ignore
    log(step=state.global_step, msg=f"Generating {N} samples for {name}")

    disable_tqdm: bool = D.rank != 0
    for i, batch in enumerate(
        tqdm(
            loader,
            desc=f"Computing metrics for {name}",
            total=len(loader),
            unit="step",
            disable=disable_tqdm,
        )
    ):
        x1 = batch["img"]
        meta = batch["meta"]

        x1 = x1.to(D.device, non_blocking=True)
        meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

        B, _, H, W = x1.shape
        x0 = torch.randn(B, 8, H // 8, W // 8, device=D.device)
        preds, _ = model.generate("px", x0=x0, meta=meta)
        preds = vae.decode(preds)

        pred_features_dino.append(dino(cp2rgb(preds)).cpu())
        real_features_dino.append(dino(cp2rgb(x1)).cpu())

        pred_features_phenom.append(phenom(preds).cpu())
        real_features_phenom.append(phenom(x1).cpu())

    D.barrier()

    # move to gpu, all_gather, move to cpu
    # we do it this way because all_gather is not supported on CPU
    def gather_cpu(x) -> np.ndarray:
        x = torch.cat(x, dim=0).to(D.device)
        x = D.gather_concat(x).cpu().numpy()
        return x[:N]  # truncate duplicates from last batch of distributed dataloader

    real_features_dino = gather_cpu(real_features_dino)
    pred_features_dino = gather_cpu(pred_features_dino)
    real_features_phenom = gather_cpu(real_features_phenom)
    pred_features_phenom = gather_cpu(pred_features_phenom)

    if D.rank == 0:
        prefix = "ema" if use_ema else "ddp"
        fd_dino = compute_fd(real_features_dino, pred_features_dino)
        fd_phenom = compute_fd(real_features_phenom, pred_features_phenom)
        cossim_phenom = compute_cossim(real_features_phenom, pred_features_phenom)
        log(
            step=state.global_step,
            data={
                f"{name}_metrics/{prefix}_fd_dinov2@{N}": fd_dino,
                f"{name}_metrics/{prefix}_fd_phemon2@{N}": fd_phenom,
                f"{name}_metrics/{prefix}_cossim_phenom2@{N}": cossim_phenom,
            },
        )

        SUBSAMPLE_N = 10000  # prdc is slow for the full dataset
        if N > SUBSAMPLE_N:
            n = SUBSAMPLE_N
            rng = np.random.default_rng(seed=42)
            idxs = rng.choice(N, size=SUBSAMPLE_N, replace=False)
            real_features_dino = real_features_dino[idxs]
            pred_features_dino = pred_features_dino[idxs]
        else:
            n = N

        prdc = compute_prdc(real_features_dino, pred_features_dino, nearest_k=5)
        prdc_preds = {
            f"{name}_metrics/{prefix}_{k}_dinov2@{n}": v for k, v in prdc.items()
        }
        log(step=state.global_step, data=prdc_preds)
    D.barrier()


# ---------------------------------------------------------------------------
# Tx evaluation helper
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_tx(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    D: Distributed,
    use_ema: bool,
) -> None:
    """Compute Tx validation loss (control -> perturbed flow matching loss)."""
    model: nn.Module = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for batch in tqdm(
        loader,
        desc="Evaluating (Tx)",
        total=len(loader),
        unit="step",
        disable=D.rank != 0,
    ):
        meta = {k: v.to(D.device, non_blocking=True) for k, v in batch["meta"].items()}
        x1 = batch["tx"].to(D.device, non_blocking=True)
        x0 = batch["tx_control"].to(D.device, non_blocking=True)

        loss = model.loss("tx", x0=x0, x1=x1, meta=meta)
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    prefix = "ema" if use_ema else "ddp"
    log(step=state.global_step, data={f"val/{prefix}_tx_loss": val_loss})
    D.barrier()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@ornamentalist.configure()
def train(
    *,
    state: TrainState,
    output_dir: str,
    D: Distributed,
    # Modality configuration (modality is determined by the model, not configurable here)
    modality: Literal["px", "tx", "joint"] = "px",
    tx_loss_weight: float = ornamentalist.Configurable[1.0],
    # Px data (required when modality is "px" or "joint")
    px_train_loader: torch.utils.data.DataLoader | None = None,
    px_val_loader: torch.utils.data.DataLoader | None = None,
    px_test_loaders: dict[str, torch.utils.data.DataLoader] | None = None,
    # Tx data (required when modality is "tx" or "joint")
    tx_train_loader: torch.utils.data.DataLoader | None = None,
    tx_val_loader: torch.utils.data.DataLoader | None = None,
    # Schedule
    num_steps: int = ornamentalist.Configurable[1_000_000],
    log_every_n_steps: int = ornamentalist.Configurable[250],
    eval_every_n_steps: int = ornamentalist.Configurable[10_000],
    ckpt_every_n_steps: int = ornamentalist.Configurable[20_000],
    metrics_every_n_steps: int = ornamentalist.Configurable[50_000],
) -> None:
    has_px = modality in ("px", "joint")
    has_tx = modality in ("tx", "joint")

    if has_px:
        assert px_train_loader is not None, "px_train_loader required for modality='px' or 'joint'"
    if has_tx:
        assert tx_train_loader is not None, "tx_train_loader required for modality='tx' or 'joint'"

    start_step = state.global_step

    # Build infinite iterators for the active modalities
    px_iter = (
        infinite_dataloader(px_train_loader, start_step=start_step, fast_resume=True)
        if has_px else None
    )
    tx_iter = (
        infinite_dataloader(tx_train_loader, start_step=start_step, fast_resume=True)
        if has_tx else None
    )

    # Only create the (expensive) VAE / colour converter when Px is active
    vae: StabilityCPEncoder | None = StabilityCPEncoder(device=D.device) if has_px else None
    cp2rgb: CellPaintConverter | None = CellPaintConverter(device=D.device) if has_px else None

    @torch.compile(fullgraph=False)
    def optim_step(step, opt, ema, ddp):
        opt.step()
        opt.zero_grad()
        ema.update(model=ddp.module, step=step)

    running_losses: dict[str, torch.Tensor] = {}
    if has_px:
        running_losses["px"] = torch.tensor(0.0, device=D.device)
    if has_tx:
        running_losses["tx"] = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()

    for step_idx in tqdm(
        range(start_step, num_steps),
        desc="Training",
        initial=start_step,
        total=num_steps,
        unit="step",
        disable=D.rank != 0,
    ):
        state.ddp.train()
        batches: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
        batch_size = 0

        # ---- Px batch ----
        if px_iter is not None:
            assert vae is not None
            px_batch = next(px_iter)
            px_meta = {k: v.to(D.device, non_blocking=True) for k, v in px_batch["meta"].items()}
            x1 = px_batch["img"].to(D.device, non_blocking=True)
            x1 = vae.encode(x1)
            x0 = torch.randn_like(x1)
            batches["px"] = (x0, x1, px_meta)
            batch_size = x0.shape[0]

        # ---- Tx batch ----
        if tx_iter is not None:
            tx_batch = next(tx_iter)
            tx_meta = {k: v.to(D.device, non_blocking=True) for k, v in tx_batch["meta"].items()}
            tx_x1 = tx_batch["tx"].to(D.device, non_blocking=True)
            tx_x0 = tx_batch["tx_control"].to(D.device, non_blocking=True)
            batches["tx"] = (tx_x0, tx_x1, tx_meta)
            if batch_size == 0:
                batch_size = tx_x0.shape[0]

        # ---- Forward / backward ----
        losses = state.ddp(batches=batches)
        total_loss = torch.tensor(0.0, device=D.device)
        for m, loss in losses.items():
            weight = tx_loss_weight if m == "tx" else 1.0
            total_loss = total_loss + weight * loss
            running_losses[m] += loss.detach() * batch_size

        num_samples += batch_size
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.ddp.parameters(), max_norm=1.0)

        state.global_step += 1
        optim_step(state.global_step, state.opt, state.ema, state.ddp)

        # ---- Logging ----
        if state.global_step % log_every_n_steps == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            throughput_per_gpu = num_samples / elapsed_time

            for m in running_losses:
                running_losses[m] = D.all_reduce(running_losses[m], op="sum")
            num_samples = D.all_reduce(num_samples, op="sum")

            throughput_total = num_samples / elapsed_time
            log_data: dict = {
                "train/throughput_total": throughput_total,
                "train/throughput_per_gpu": throughput_per_gpu,
                "train/grad_norm": grad_norm,
            }
            for m in running_losses:
                log_data[f"train/{m}_loss"] = (running_losses[m] / num_samples).item()
                running_losses[m] = torch.tensor(0.0, device=D.device)

            # Convenience "train/loss" key = total weighted loss
            total = sum(
                (tx_loss_weight if m == "tx" else 1.0) * log_data[f"train/{m}_loss"]
                for m in losses
            )
            log_data["train/loss"] = total
            num_samples = torch.tensor(0, device=D.device)
            log(step=state.global_step, data=log_data)

        # ---- Checkpointing ----
        if state.global_step % ckpt_every_n_steps == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            log(step=state.global_step, msg=f"Saving checkpoint to {ckpt_dir}")
            state.save_latest_ckpt(dir=ckpt_dir)

        # ---- Evaluation ----
        if state.global_step % eval_every_n_steps == 0:
            if has_px and px_val_loader is not None:
                assert vae is not None and cp2rgb is not None
                visualise_phenomics(
                    state=state,
                    loader=px_val_loader,
                    output_dir=output_dir,
                    vae=vae,
                    cp2rgb=cp2rgb,
                    D=D,
                    use_ema=True,
                )
                evaluate_px(state=state, vae=vae, loader=px_val_loader, D=D, use_ema=False)
                evaluate_px(state=state, vae=vae, loader=px_val_loader, D=D, use_ema=True)

            if has_tx and tx_val_loader is not None:
                evaluate_tx(state=state, loader=tx_val_loader, D=D, use_ema=False)
                evaluate_tx(state=state, loader=tx_val_loader, D=D, use_ema=True)

        # ---- Expensive metrics (Px only) ----
        if state.global_step % metrics_every_n_steps == 0:
            if has_px and px_test_loaders is not None:
                assert vae is not None and cp2rgb is not None
                for name, test_loader in px_test_loaders.items():
                    compute_phenomics_metrics(
                        state=state,
                        name=name,
                        vae=vae,
                        cp2rgb=cp2rgb,
                        loader=test_loader,
                        D=D,
                        use_ema=True,
                    )

    log(step=state.global_step, msg="Training complete")
    return


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by eval.py, gen_anax.py, etc.)
# ---------------------------------------------------------------------------
evaluate = evaluate_px
visualise = visualise_phenomics
compute_metrics = compute_phenomics_metrics
