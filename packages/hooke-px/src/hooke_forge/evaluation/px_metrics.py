import os

import numpy as np
import torch
import wandb
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

from hooke_forge.data.dataset import CellPaintConverter
from hooke_forge.evaluation.metrics import compute_cossim, compute_fd, compute_prdc
from hooke_forge.training.state import TrainState, log
from hooke_forge.utils.distributed import Distributed, rank_zero
from hooke_forge.utils.encoders import DINOv2Detector, Phenom2Detector, StabilityCPEncoder


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
    compute_phenom: bool = False,
) -> None:
    model: nn.Module = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    dino = DINOv2Detector(device=D.device)
    real_features_dino = []
    pred_features_dino = []
    if compute_phenom:
        phenom = Phenom2Detector(device=D.device)
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
        if compute_phenom:
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
    if compute_phenom:
        real_features_phenom = gather_cpu(real_features_phenom)
        pred_features_phenom = gather_cpu(pred_features_phenom)

    if D.rank == 0:
        prefix = "ema" if use_ema else "ddp"
        fd_dino = compute_fd(real_features_dino, pred_features_dino)
        data = {
            f"{name}_metrics/{prefix}_fd_dinov2@{N}": fd_dino,
        }
        if compute_phenom:
            fd_phenom = compute_fd(real_features_phenom, pred_features_phenom)
            cossim_phenom = compute_cossim(real_features_phenom, pred_features_phenom)
            data[f"{name}_metrics/{prefix}_fd_phemon2@{N}"] = fd_phenom
            data[f"{name}_metrics/{prefix}_cossim_phenom2@{N}"] = cossim_phenom
        log(
            step=state.global_step,
            data=data,
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
