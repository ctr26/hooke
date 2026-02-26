import dataclasses
import json
import logging
import os
import re
import tempfile

import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from hooke_forge.model.tokenizer import DataFrameTokenizer
from hooke_forge.utils.distributed import rank_zero
from hooke_forge.utils.ema import KarrasEMA

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
