from typing import Any

import torch
import lightning.pytorch as pl

from hooke_tx.architecture.binder import TxGenBinder


SCHEDULER_DICT = {
    "cosine": (torch.optim.lr_scheduler.CosineAnnealingLR, "T_max"),
    "linear": (torch.optim.lr_scheduler.LinearLR, "total_iters"),
}


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        covariates: dict[str, list[str | float]],
        data_dim: int,
        model_args: dict[str, dict[str, Any]],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        max_epochs: int = 100,
        warmup_epochs: int = 0,
        gradient_accumulation_steps: int = 1,
        scheduler_args: dict[str, Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler_args = scheduler_args

        self.architecture = TxGenBinder(
            covariates=covariates,
            data_dim=data_dim,
            **model_args
        )

    def forward(self, batch: dict) -> torch.Tensor:
        return self.architecture(batch)

    def training_step(self, batch, batch_idx):
        loss = self.architecture(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.architecture(batch)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.architecture.generate(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        schedulers = []
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            schedulers.append(warmup)

        if self.scheduler_args is not None:
            scheduler_args = {k: v for k, v in self.scheduler_args.items() if k != "type"}
            scheduler_type = self.scheduler_args["type"]
            scheduler_cls, epoch_param = SCHEDULER_DICT[scheduler_type]

            # When chained with warmup, make main scheduler span remaining epochs
            if schedulers:
                scheduler_args[epoch_param] = self.max_epochs - self.warmup_epochs
            main_scheduler = scheduler_cls(optimizer, **scheduler_args)
            schedulers.append(main_scheduler)

        if not schedulers:
            return [optimizer]

        if len(schedulers) == 1:
            return [optimizer], [schedulers[0]]

        # Chain warmup then main via SequentialLR (milestones = epochs until switch)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[self.warmup_epochs]
        )
        return [optimizer], [scheduler]