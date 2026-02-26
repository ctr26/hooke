import torch
from torch import nn
from tqdm import tqdm

from hooke_forge.training.state import TrainState, log
from hooke_forge.utils.distributed import Distributed


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
