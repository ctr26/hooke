import os
import time
from typing import Literal

import ornamentalist
import torch
from tqdm import tqdm

from hooke_forge.data.dataset import CellPaintConverter
from hooke_forge.evaluation.px_metrics import (
    compute_phenomics_metrics,
    evaluate_px,
    visualise_phenomics,
)
from hooke_forge.evaluation.tx_metrics import evaluate_tx
from hooke_forge.training.state import TrainState, log
from hooke_forge.utils.distributed import Distributed
from hooke_forge.utils.encoders import StabilityCPEncoder
from hooke_forge.utils.infinite_dataloader import infinite_dataloader


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
    px_iter = infinite_dataloader(px_train_loader, start_step=start_step, fast_resume=True) if has_px else None
    tx_iter = infinite_dataloader(tx_train_loader, start_step=start_step, fast_resume=True) if has_tx else None

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
            total = sum((tx_loss_weight if m == "tx" else 1.0) * log_data[f"train/{m}_loss"] for m in losses)
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
