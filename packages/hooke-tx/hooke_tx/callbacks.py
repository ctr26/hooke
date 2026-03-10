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
    Save frequency is synced with eval: use every_n_train_steps (step-based) or every_n_epochs (epoch-based).
    """

    def __init__(
        self,
        decay: float = 0.999,
        save_checkpoint: bool = False,
        dirpath: str | None = None,
        filename: str = "{epoch}",
        every_n_epochs: int | None = None,
        every_n_train_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.save_checkpoint = save_checkpoint
        self.dirpath = dirpath or "."
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
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

    def _save_flag(self, trainer: pl.Trainer) -> bool:
        if self.every_n_train_steps is not None:
            return trainer.global_step > 0 and trainer.global_step % self.every_n_train_steps == 0
        
        if self.every_n_epochs is not None:
            return (trainer.current_epoch + 1) % self.every_n_epochs == 0
        
        return True

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.save_checkpoint or not trainer.is_global_zero or self._ema is None:
            return
        if not self._save_flag(trainer):
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
    metrics_args: dict[str, Any],
    trainer_args: dict[str, Any],
    eval_args: dict[str, Any],
    checkpoint_args: dict[str, Any],
) -> tuple[list[Callback], dict[str, Any]]:
    """Create training callbacks from config. Pops ema, eval from trainer_args and types, enable from checkpoint_args.

    Returns:
        Tuple of (callbacks, logging_kwargs). Pass logging_kwargs to Trainer for val_check_interval / check_val_every_n_epoch.

    Note:
        With val_check_interval (step-based eval), ModelCheckpoint saves on train batch end when
        global_step % every_n_train_steps == 0, while EvalCallback/EMACallback run on validation_epoch_end.
        Both fire at the same step boundaries (validation is triggered at those steps), so checkpoints
        and eval metrics stay aligned.
    """
    ema_config = trainer_args.pop("ema", None) or {}
    ema_enable = ema_config.get("enable", False)

    eval_base = eval_args.pop("base", True)
    eval_ema = eval_args.pop("ema", False)

    if eval_args.get("val_check_interval") is not None:
        eval_args.pop("check_val_every_n_epoch")

    val_check_interval = eval_args.get("val_check_interval")
    
    logging_kwargs = (
        {"val_check_interval": val_check_interval, "check_val_every_n_epoch": 1}
        if val_check_interval is not None
        else {"check_val_every_n_epoch": eval_args.get("check_val_every_n_epoch", 1)}
    )

    if eval_ema and not ema_enable:
        raise ValueError("trainer.eval.ema is True but ema.enable is False")

    checkpoint_types = checkpoint_args.pop("types", None) or {}
    checkpoint_base = checkpoint_types.get("base", True)
    checkpoint_ema = checkpoint_types.get("ema", False)

    checkpoint_enable = checkpoint_args.pop("enable", False)
    every_n_evals = checkpoint_args.pop("every_n_evals", 1)
    
    if checkpoint_ema and not ema_enable:
        raise ValueError("checkpoint.types.ema is True but ema.enable is False")

    # Sync checkpoint frequency with eval; every_n_evals saves every N evals
    if val_check_interval is not None:
        ckpt_freq = {
            "every_n_train_steps": val_check_interval * every_n_evals,
            "every_n_epochs": None,
        }
    else:
        check_val_every_n_epoch = eval_args.get("check_val_every_n_epoch", 1)
        ckpt_freq = {
            "every_n_train_steps": None,
            "every_n_epochs": check_val_every_n_epoch * every_n_evals,
        }

    callbacks: list[Callback] = []
    ema_callback = None

    if ema_enable:
        ema_callback = EMACallback(
            decay=ema_config.get("decay", 0.999),
            save_checkpoint=checkpoint_enable and checkpoint_ema,
            dirpath=checkpoint_args.get("dirpath"),
            filename=checkpoint_args.get("filename", "{epoch}"),
            **ckpt_freq,
        )
        callbacks.append(ema_callback)

    callbacks.append(
        EvalCallback(
            metrics_args=metrics_args,
            eval_base=eval_base,
            eval_ema=eval_ema,
            ema_callback=ema_callback,
        )
    )

    if checkpoint_enable and checkpoint_base:
        callbacks.append(
            ModelCheckpoint(
                **checkpoint_args,
                **ckpt_freq,
            )
        )

    return callbacks, logging_kwargs
