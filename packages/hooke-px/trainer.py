import dataclasses
import logging
import math
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

from dataset import CellPaintConverter, StabilityCPEncoder
from model import DiTWrapper
from utils.distributed import Distributed, rank_zero
from utils.ema import KarrasEMA
from utils.evaluation import DINOv2Detector, compute_fd, compute_prdc
from utils.infinite_dataloader import infinite_dataloader
from utils.profiler import get_profiler

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


@rank_zero()
def log(*, step: int, msg: str | None = None, data: dict | None = None):
    if msg is not None:
        _log.info(f"[{step=}] {msg}")
    if data is not None:
        wandb.log(data, step=step)


@dataclasses.dataclass
class TrainState:
    ddp: DDP  # Distributed wrapper around DiT model
    ema: KarrasEMA  # EMA wrapper around DiT model
    opt: torch.optim.Optimizer
    global_step: int

    def _to_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "ddp": self.ddp.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }

    @rank_zero()
    def save_ckpt(self, path: str) -> None:
        dir = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dir, suffix=".tmp") as f:
            temp_path = f.name
            torch.save(self._to_dict(), temp_path)
        os.rename(temp_path, path)

    def load_ckpt(self, path: str, device: torch.device) -> None:
        state = torch.load(path, weights_only=True, map_location=device)
        self.global_step = state["global_step"]
        self.ddp.load_state_dict(state["ddp"])
        self.ema.load_state_dict(state["ema"])
        self.opt.load_state_dict(state["opt"])

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


@ornamentalist.configure()
def guided_prediction(
    model: DiTWrapper,
    x, t, z, y, e, c,
    cfg: float = ornamentalist.Configurable[1.0],
) -> torch.Tensor:  # fmt: off
    if t.ndim == 0:  # the ODE solver gives scalar t
        t = t.expand(x.shape[0])

    if cfg == 0.0:  # unconditional
        return model(x=x, t=t, z=z, y=None, e=e, c=c)
    if cfg == 1.0:  # conditional
        return model(x=x, t=t, z=z, y=y, e=e, c=c)

    pred_cond = model(x=x, t=t, z=z, y=y, e=e, c=c)
    pred_null = model(x=x, t=t, z=z, y=None, e=e, c=c)
    return pred_null + cfg * (pred_cond - pred_null)


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate(
    model: DiTWrapper,  # DiT, maps x,cond -> velocity
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from N(0, I)
    y1: torch.Tensor,  # shape (B,) - perturbation condition
    z1: torch.Tensor,  # shape (B,) - zoom condition
    e1: torch.Tensor,  # shape (B,) - experiment condition
    c1: torch.Tensor,  # shape (B,) - cell type condition
) -> tuple[torch.Tensor, int]:
    """Generate a sample with the dopri5 probability flow ODE solver."""
    nfe = 0  # NB, if using guidance, the true nfe is this * 2

    def forward_fn(t, x):
        nonlocal nfe
        nfe += 1
        return guided_prediction(model, x=x, t=t, z=z1, y=y1, e=e1, c=c1)

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
@ornamentalist.configure()
def compute_loss(
    model: DDP,  # DDP wrapper around DiT, maps t,x,y,z,e,c -> velocity
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from source distribution
    x1: torch.Tensor,  # shape (B, C, H, W), sampled from target distribution
    y1: torch.Tensor,  # shape (B,) - [0,perturbations)
    z1: torch.Tensor,  # shape (B,) - zoom condition
    e1: torch.Tensor,  # shape (B,) - experiment condition
    c1: torch.Tensor,  # shape (B,) - cell type condition
) -> torch.Tensor:
    t = torch.rand_like(y1, dtype=torch.float32)  # shape (B,) - [0,1)
    t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

    xt = torch.lerp(x0, x1, t_)
    ut = x1 - x0

    vt = model(x=xt, t=t, z=z1, y=y1, e=e1, c=c1)
    return torch.nn.functional.mse_loss(vt, ut)


@torch.inference_mode()
def evaluate(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    vae: StabilityCPEncoder,
    D: Distributed,
) -> None:
    model = state.ddp.module
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
        y1 = batch["perturbation_id"]
        exp = batch["experiment_id"]
        cell_type = batch["cell_type_id"]
        zoom = batch["zoom_id"]

        x1 = x1.to(D.device, non_blocking=True)
        y1 = y1.to(D.device, non_blocking=True)
        exp = exp.to(D.device, non_blocking=True)
        cell_type = cell_type.to(D.device, non_blocking=True)
        zoom = zoom.to(D.device, non_blocking=True)

        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)

        loss = compute_loss(
            model=model,
            x0=x0,
            x1=x1,
            y1=y1,
            z1=zoom,
            e1=exp,
            c1=cell_type,
        )
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    log(step=state.global_step, data={"val/loss": val_loss})
    D.barrier()
    return


@rank_zero()
@torch.inference_mode()
@ornamentalist.configure()
def visualise(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    vae: StabilityCPEncoder,
    cp2rgb: CellPaintConverter,
    D: Distributed,
    use_ema: bool = ornamentalist.Configurable[True],
    num_samples: int = ornamentalist.Configurable[16],
):
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    batch = next(iter(loader))
    x1 = batch["img"]
    y1 = batch["perturbation_id"]
    exp = batch["experiment_id"]
    cell_type = batch["cell_type_id"]
    zoom = batch["zoom_id"]

    x1 = x1.to(D.device, non_blocking=True)
    y1 = y1.to(D.device, non_blocking=True)
    exp = exp.to(D.device, non_blocking=True)
    cell_type = cell_type.to(D.device, non_blocking=True)
    zoom = zoom.to(D.device, non_blocking=True)

    x1 = vae.encode(x1)
    x0 = torch.randn_like(x1)
    preds, nfe = generate(
        model=model,
        x0=x0,
        y1=y1,
        z1=zoom,
        e1=exp,
        c1=cell_type,
    )
    preds = vae.decode(preds)
    preds = cp2rgb(preds)[:num_samples]  # uint8 [B, 3, H, W]
    preds = preds.to(torch.float32) / 255  # save_image wants float32

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    save_path = os.path.join(output_dir, f"samples/{state.global_step}.png")
    save_image(preds, save_path)
    log(
        step=state.global_step,
        data={"val/images": wandb.Image(save_path), "val/nfe": nfe},
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
    use_ema: bool = ornamentalist.Configurable[True],
) -> None:
    model: DiTWrapper = state.ema.module if use_ema else state.ddp.module  # type: ignore
    model.eval()

    detector = DINOv2Detector()

    real_features = []
    pred_features = []

    N = len(loader.dataset)  # type: ignore
    log(step=state.global_step, msg=f"Generating {N} samples for {name}")

    disable_tqdm: bool = D.rank != 0
    for i, batch in enumerate(
        tqdm(
            loader,
            desc=f"Computing FD metrics for {name}",
            total=len(loader),
            unit="step",
            disable=disable_tqdm,  # only show from one process
        )
    ):
        x1 = batch["img"]
        y1 = batch["perturbation_id"]

        exp = batch["experiment_id"]
        cell_type = batch["cell_type_id"]
        zoom = batch["zoom_id"]

        x1 = x1.to(D.device, non_blocking=True)
        y1 = y1.to(D.device, non_blocking=True)
        exp = exp.to(D.device, non_blocking=True)
        cell_type = cell_type.to(D.device, non_blocking=True)
        zoom = zoom.to(D.device, non_blocking=True)
        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)

        preds, _ = generate(
            model=model,
            x0=x0,
            y1=y1,
            z1=zoom,
            e1=exp,
            c1=cell_type,
        )
        preds = vae.decode(preds)
        pred_features.append(detector(cp2rgb(preds)).cpu())
        real_features.append(detector(cp2rgb(x1)).cpu())

    D.barrier()

    # move to gpu, all_gather, move to cpu
    # we do it this way because all_gather is not supported on CPU
    real_features = torch.cat(real_features, dim=0).to(D.device)
    real_features = D.gather_concat(real_features).cpu().numpy()

    pred_features = torch.cat(pred_features, dim=0).to(D.device)
    pred_features = D.gather_concat(pred_features).cpu().numpy()

    # truncate duplicated samples in last batch from distributed dataloader
    real_features = real_features[:N]
    pred_features = pred_features[:N]

    if D.rank == 0:
        fd = compute_fd(real_features, pred_features)
        log(step=state.global_step, data={f"{name}_metrics/fd@{N}": fd})

        SUBSAMPLE_N = 10000  # prdc is slow for the full dataset
        n_samples = pred_features.shape[0]
        if n_samples > SUBSAMPLE_N:
            n = SUBSAMPLE_N
            rng = np.random.default_rng(seed=42)
            idxs = rng.choice(n_samples, size=SUBSAMPLE_N, replace=False)
            real_features = real_features[idxs]
            pred_features = pred_features[idxs]
        else:
            n = n_samples

        prdc = compute_prdc(real_features, pred_features, nearest_k=5)
        prdc_preds = {f"{name}_metrics/{k}@{n}": v for k, v in prdc.items()}
        log(step=state.global_step, data=prdc_preds)
    D.barrier()


@ornamentalist.configure()
def lr_schedule(
    optimizer: torch.optim.Optimizer,
    global_step: int,
    warmup_steps: int = ornamentalist.Configurable[5_000],
    constant_steps: int = ornamentalist.Configurable[45_000],
    max_lr: float = ornamentalist.Configurable[1e-4],
) -> float:
    """Inverse square-root decay scheduler. Returns the new learning rate
    at the current step, and sets the optimizer to that lr as a side effect."""
    decay_start = warmup_steps + constant_steps
    new_lr = max_lr

    new_lr *= min(global_step / warmup_steps, 1.0)  # apply warmup
    new_lr /= math.sqrt(max(global_step / decay_start, 1.0))  # apply decay

    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    return new_lr


@ornamentalist.configure()
def train(
    *,
    state: TrainState,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loaders: dict[str, torch.utils.data.DataLoader],
    output_dir: str,
    D: Distributed,
    num_steps: int = ornamentalist.Configurable[250_000],
    log_every_n_steps: int = ornamentalist.Configurable[250],
    eval_every_n_steps: int = ornamentalist.Configurable[10_000],
    ckpt_every_n_steps: int = ornamentalist.Configurable[50_000],
    metrics_every_n_steps: int = ornamentalist.Configurable[50_000],
) -> None:
    prof = get_profiler(return_dummy=D.rank != 0, save_dir=output_dir)
    start_step = state.global_step
    loader = infinite_dataloader(train_loader, start_step=start_step, fast_resume=True)

    vae = StabilityCPEncoder(device=D.device)
    cp2rgb = CellPaintConverter(device=D.device)

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()
    prof.start()
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
        y1 = batch["perturbation_id"]
        exp = batch["experiment_id"]
        cell_type = batch["cell_type_id"]
        zoom = batch["zoom_id"]

        x1 = x1.to(D.device, non_blocking=True)
        y1 = y1.to(D.device, non_blocking=True)
        exp = exp.to(D.device, non_blocking=True)
        cell_type = cell_type.to(D.device, non_blocking=True)
        zoom = zoom.to(D.device, non_blocking=True)

        x1 = vae.encode(x1)
        x0 = torch.randn_like(x1)
        loss = compute_loss(
            model=state.ddp,
            x0=x0,
            x1=x1,
            y1=y1,
            z1=zoom,
            e1=exp,
            c1=cell_type,
        )
        running_loss += loss.detach() * x0.shape[0]  # total batch loss
        num_samples += x0.shape[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.ddp.parameters(), max_norm=1.0)

        state.global_step += 1
        lr = lr_schedule(optimizer=state.opt, global_step=state.global_step)
        state.opt.step()
        state.opt.zero_grad()
        state.ema.update(model=state.ddp.module, step=state.global_step)
        prof.step()

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
                    "train/lr": lr,
                },
            )

        if state.global_step % eval_every_n_steps == 0:
            visualise(
                state=state,
                loader=val_loader,
                output_dir=output_dir,
                vae=vae,
                cp2rgb=cp2rgb,
                D=D,
            )
            evaluate(state=state, vae=vae, loader=val_loader, D=D)

        if state.global_step % metrics_every_n_steps == 0:
            for name, loader in test_loaders.items():
                compute_metrics(
                    state=state, name=name, vae=vae, cp2rgb=cp2rgb, loader=loader, D=D
                )

        if state.global_step % ckpt_every_n_steps == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            log(step=state.global_step, msg=f"Saving checkpoint to {ckpt_dir}")
            state.save_latest_ckpt(dir=ckpt_dir)

    log(step=state.global_step, msg="Training complete")
    return
