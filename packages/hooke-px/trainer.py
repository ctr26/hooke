import dataclasses
import json
import logging
import os
import re
import tempfile
import time

import numpy as np
import ornamentalist
import torch
import torchdiffeq
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm

from adaptor import DataFrameTokenizer
from dataset import CellPaintConverter, StabilityCPEncoder
from model import DiTWrapper
from utils.distributed import Distributed, rank_zero
from utils.ema import KarrasEMA
from utils.evaluation import (
    DINOv2Detector,
    Phenom2Detector,
    compute_cossim,
    compute_fd,
    compute_prdc,
)
from utils.infinite_dataloader import infinite_dataloader
from utils.profiler import get_profiler

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
    ddp: DDP  # Distributed wrapper around DiT model
    ema: KarrasEMA  # EMA wrapper around DiT model
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

    def load_ckpt(self, path: str, device: torch.device) -> None:
        state = torch.load(path, weights_only=False, map_location=device)
        self.global_step = state["global_step"]
        self.ddp.load_state_dict(state["ddp"])
        self.ema.load_state_dict(state["ema"])
        self.opt.load_state_dict(state["opt"])
        if "tokenizer" in state:
            self.tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    def save_latest_ckpt(self, dir: str) -> None:
        path = os.path.join(dir, f"step_{self.global_step}.ckpt")
        self.save_ckpt(path)

    def load_latest_ckpt(self, dir: str, device: torch.device) -> None:
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
        self.load_ckpt(path, device)


# @ornamentalist.configure()
def guided_prediction(
    model: DiTWrapper,
    x,
    t,
    meta: dict[str, torch.Tensor],
    cfg: float = 1.0,
) -> torch.Tensor:  # fmt: off
    if t.ndim == 0:  # the ODE solver gives scalar t
        t = t.expand(x.shape[0])

    if cfg == 0.0:  # unconditional
        force_drop = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        return model(x=x, t=t, meta=meta, force_drop_rec_conc=force_drop)
    if cfg == 1.0:  # conditional
        return model(x=x, t=t, meta=meta, force_drop_rec_conc=None)

    pred_cond = model(x=x, t=t, meta=meta, force_drop_rec_conc=None)
    force_drop = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
    pred_null = model(x=x, t=t, meta=meta, force_drop_rec_conc=force_drop)
    return pred_null + cfg * (pred_cond - pred_null)


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate(
    model: DiTWrapper,  # DiT, maps x,cond -> velocity
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from N(0, I)
    meta1: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """Generate a sample with the dopri5 probability flow ODE solver."""
    nfe = 0  # NB, if using guidance, the true nfe is this * 2

    def forward_fn(t, x):
        nonlocal nfe
        nfe += 1
        return guided_prediction(model, x=x, t=t, meta=meta1)

    traj = torchdiffeq.odeint(
        forward_fn,
        x0,
        torch.linspace(0, 1, 2, device=x0.device),
        method="dopri5",
        rtol=1e-5,
        atol=1e-5,
    )
    return traj[-1], nfe  # type: ignore


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_loss(
    model: DDP,  # DDP wrapper around DiT, maps t,x,y,z,e,c -> velocity
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from source distribution
    x1: torch.Tensor,  # shape (B, C, H, W), sampled from target distribution
    meta1: dict[str, torch.Tensor],
) -> torch.Tensor:
    t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)  # (B,) - [0,1)
    t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

    xt = torch.lerp(x0, x1, t_)
    ut = x1 - x0

    vt = model(x=xt, t=t, meta=meta1)
    return torch.nn.functional.mse_loss(vt, ut)


@torch.inference_mode()
def evaluate(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    vae: StabilityCPEncoder,
    D: Distributed,
    use_ema: bool,
) -> None:
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for batch in tqdm(
        loader,
        desc="Evaluating",
        total=len(loader),
        unit="step",
        disable=D.rank != 0,  # only show from one process
    ):
        x1 = batch["img"]
        meta = batch["meta"]

        x1 = x1.to(D.device, non_blocking=True)
        meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)

        loss = compute_loss(
            model=model,  # type: ignore
            x0=x0,
            x1=x1,
            meta1=meta,
        )
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    prefix = "ema" if use_ema else "ddp"
    log(step=state.global_step, data={f"val/{prefix}_loss": val_loss})
    D.barrier()
    return


@rank_zero()
@torch.inference_mode()
def visualise(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    vae: StabilityCPEncoder,
    cp2rgb: CellPaintConverter,
    D: Distributed,
    use_ema: bool,
):
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    batch = next(iter(loader))
    x1 = batch["img"]
    meta = batch["meta"]

    x1 = x1.to(D.device, non_blocking=True)
    meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

    x1 = vae.encode(x1)
    x0 = torch.randn_like(x1)
    preds, nfe = generate(
        model=model,
        x0=x0,
        meta1=meta,
    )
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
def compute_metrics(
    *,
    state: TrainState,
    name: str,
    loader: torch.utils.data.DataLoader,
    vae: StabilityCPEncoder,
    cp2rgb: CellPaintConverter,
    D: Distributed,
    use_ema: bool,
) -> None:
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
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
            disable=disable_tqdm,  # only show from one process
        )
    ):
        x1 = batch["img"]
        meta = batch["meta"]

        x1 = x1.to(D.device, non_blocking=True)
        meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

        B, _, H, W = x1.shape
        x0 = torch.randn(B, 8, H // 8, W // 8, device=D.device)
        preds, _ = generate(
            model=model,
            x0=x0,
            meta1=meta,
        )
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


@ornamentalist.configure()
def train(
    *,
    state: TrainState,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loaders: dict[str, torch.utils.data.DataLoader],
    output_dir: str,
    D: Distributed,
    num_steps: int = ornamentalist.Configurable[1_000_000],
    log_every_n_steps: int = ornamentalist.Configurable[250],
    eval_every_n_steps: int = ornamentalist.Configurable[10_000],
    ckpt_every_n_steps: int = ornamentalist.Configurable[20_000],
    metrics_every_n_steps: int = ornamentalist.Configurable[50_000],
) -> None:
    # prof = get_profiler(return_dummy=D.rank != 0, save_dir=output_dir)
    start_step = state.global_step
    loader = infinite_dataloader(train_loader, start_step=start_step, fast_resume=True)

    vae = StabilityCPEncoder(device=D.device)
    cp2rgb = CellPaintConverter(device=D.device)

    @torch.compile(fullgraph=False)
    def step(step, opt, ema, ddp):  # compiling the optimiser and ema step is helpful
        opt.step()
        opt.zero_grad()
        ema.update(model=ddp.module, step=step)

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()
    # prof.start()
    for _, batch in tqdm(
        zip(range(start_step, num_steps), loader),
        desc="Training",
        initial=start_step,
        total=num_steps,
        unit="step",
        disable=D.rank != 0,  # only show from one process
    ):
        state.ddp.train()
        x1 = batch["img"]
        meta = batch["meta"]

        x1 = x1.to(D.device, non_blocking=True)
        meta = {k: v.to(D.device, non_blocking=True) for k, v in meta.items()}

        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)
        loss = compute_loss(
            model=state.ddp,
            x0=x0,
            x1=x1,
            meta1=meta,
        )
        running_loss += loss.detach() * x0.shape[0]  # total batch loss
        num_samples += x0.shape[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.ddp.parameters(), max_norm=1.0)

        state.global_step += 1
        step(state.global_step, state.opt, state.ema, state.ddp)
        # prof.step()

        if state.global_step % log_every_n_steps == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            throughput_per_gpu = num_samples / elapsed_time

            running_loss = D.all_reduce(running_loss, op="sum")
            num_samples = D.all_reduce(num_samples, op="sum")

            throughput_total = num_samples / elapsed_time
            loss = (running_loss / num_samples).item()
            running_loss = torch.tensor(0.0, device=D.device)
            num_samples = torch.tensor(0, device=D.device)

            log(
                step=state.global_step,
                data={
                    "train/loss": loss,
                    "train/throughput_total": throughput_total,
                    "train/throughput_per_gpu": throughput_per_gpu,
                    "train/grad_norm": grad_norm,
                },
            )
        if state.global_step % ckpt_every_n_steps == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            log(step=state.global_step, msg=f"Saving checkpoint to {ckpt_dir}")
            state.save_latest_ckpt(dir=ckpt_dir)

        if state.global_step % eval_every_n_steps == 0:
            visualise(
                state=state,
                loader=val_loader,
                output_dir=output_dir,
                vae=vae,
                cp2rgb=cp2rgb,
                D=D,
                use_ema=True,
            )
            evaluate(state=state, vae=vae, loader=val_loader, D=D, use_ema=False)
            evaluate(state=state, vae=vae, loader=val_loader, D=D, use_ema=True)

        if state.global_step % metrics_every_n_steps == 0:
            for name, loader in test_loaders.items():
                compute_metrics(
                    state=state,
                    name=name,
                    vae=vae,
                    cp2rgb=cp2rgb,
                    loader=loader,
                    D=D,
                    use_ema=True,
                )

    log(step=state.global_step, msg="Training complete")
    return
