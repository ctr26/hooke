"""Training callbacks: EMA, EvalCallback factory, and related utilities."""

from __future__ import annotations

import os
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from torch_ema import ExponentialMovingAverage

from hooke_tx.eval.evaluation import EvalCallback


class EMACallback(Callback):
    """Maintains EMA of model parameters. Updates after each training batch.
    Optionally saves EMA to a separate checkpoint file when checkpoint.types.ema is True.
    """

    def __init__(
        self,
        decay: float = 0.999,
        save_checkpoint: bool = False,
        dirpath: str | None = None,
        filename: str = "{epoch}",
        every_n_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.save_checkpoint = save_checkpoint
        self.dirpath = dirpath or "."
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self._ema: ExponentialMovingAverage | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if stage == "fit":
            self._ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._ema is not None:
            self._ema.to(pl_module.device)
            self._ema.update(pl_module.parameters())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.save_checkpoint or not trainer.is_global_zero or self._ema is None:
            return

        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        self._ema.store(pl_module.parameters())
        self._ema.copy_to(pl_module.parameters())
        try:
            try:
                formatted = self.filename.format(
                    epoch=trainer.current_epoch + 1,
                    step=trainer.global_step,
                )
            except KeyError:
                formatted = f"{trainer.current_epoch + 1}"
            path = os.path.join(self.dirpath, f"{formatted}_ema.ckpt")
            trainer.save_checkpoint(path)
        finally:
            self._ema.restore(pl_module.parameters())

    def state_dict(self) -> dict:
        if self._ema is None:
            return {}
        return {"ema": self._ema.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._ema is not None and "ema" in state_dict:
            self._ema.load_state_dict(state_dict["ema"])

    @property
    def ema(self) -> ExponentialMovingAverage | None:
        return self._ema


def create_callbacks(
    metrics_config: dict[str, Any],
    trainer_args: dict[str, Any],
    checkpoint_args: dict[str, Any],
) -> list[Callback]:
    """Create training callbacks from config. Pops ema, eval from trainer_args and types, enable from checkpoint_args."""
    ema_config = trainer_args.pop("ema", None) or {}
    ema_enable = ema_config.get("enable", False)

    eval_config = trainer_args.pop("eval", None) or {}
    eval_standard = eval_config.get("standard", True)
    eval_ema = eval_config.get("ema", False)
    if eval_ema and not ema_enable:
        raise ValueError("trainer.eval.ema is True but ema.enable is False")

    checkpoint_types = checkpoint_args.pop("types", None) or {}
    checkpoint_standard = checkpoint_types.get("standard", True)
    checkpoint_ema = checkpoint_types.get("ema", False)
    checkpoint_enable = checkpoint_args.pop("enable", False)
    if checkpoint_ema and not ema_enable:
        raise ValueError("checkpoint.types.ema is True but ema.enable is False")

    callbacks: list[Callback] = []
    ema_callback = None

    if ema_enable:
        ema_callback = EMACallback(
            decay=ema_config.get("decay", 0.999),
            save_checkpoint=checkpoint_enable and checkpoint_ema,
            dirpath=checkpoint_args.get("dirpath"),
            filename=checkpoint_args.get("filename", "{epoch}"),
            every_n_epochs=checkpoint_args.get("every_n_epochs", 5),
        )
        callbacks.append(ema_callback)

    callbacks.append(
        EvalCallback(
            metrics_config=metrics_config,
            eval_standard=eval_standard,
            eval_ema=eval_ema,
            ema_callback=ema_callback,
        )
    )

    if checkpoint_enable and checkpoint_standard:
        callbacks.append(ModelCheckpoint(**checkpoint_args))

    return callbacks
