"""Evaluation: compare prediction and target DataFrames and compute metrics."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import gc
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

import wandb

import lightning.pytorch as pl
from lightning.pytorch import Callback

from hooke_tx.data.constants import CELL_TYPE, EXPERIMENT, PERTURBATIONS
from hooke_tx.eval.inference import run_inference

from scipy.stats import pearsonr
from hooke_tx.eval.metrics.distributional import compute_e_distance


METRIC_FUNCTION_DICT = {
    "aggr": {
        "mae": lambda x, y: np.mean(np.abs(x.mean(axis=0) - y.mean(axis=0))),
        "mse": lambda x, y: np.mean((x.mean(0) - y.mean(0)) ** 2),
        "pearson": lambda x, y: pearsonr(x.mean(0), y.mean(0))[0],
    },
    "dist": {
        "mmd": lambda pred_expr, target_expr: compute_e_distance(pred_expr, target_expr),
    },
}


class EvalCallback(Callback):
    """Runs inference + evaluation metrics at the end of each validation epoch and logs them."""
    def __init__(
        self,
        metrics_args: dict[str, Any],
        eval_standard: bool = True,
        eval_ema: bool = False,
        ema_callback: Callback | None = None,
    ) -> None:
        super().__init__()
        self.metrics_args = metrics_args
        self.ema_callback = ema_callback

        self.eval_models = []
        if eval_standard:
            self.eval_models.append("standard")
        if eval_ema:
            self.eval_models.append("ema")

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        if not self.eval_models:
            return

        val_dataloaders = trainer.val_dataloaders

        if val_dataloaders is None:
            return

        elif isinstance(val_dataloaders, dict) and len(val_dataloaders) > 0:
            pass

        elif isinstance(val_dataloaders, DataLoader):
            val_dataloaders = {"eval": val_dataloaders}

        else:
            raise ValueError(f"Invalid val_dataloader format: {val_dataloaders}")

        device = trainer.strategy.root_device

        all_metrics: dict[str, float] = {}

        # Train and eval loss at eval time, per model type
        train_loss = None
        if trainer.is_global_zero:
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss is not None:
                train_loss = float(train_loss.item() if hasattr(train_loss, "item") else train_loss)
        
        standard_eval_loss = None
        if trainer.is_global_zero:
            std_eval_loss = trainer.callback_metrics.get("standard/eval/loss")
            if std_eval_loss is not None:
                std_eval_loss = float(std_eval_loss.item() if hasattr(std_eval_loss, "item") else std_eval_loss)

        for eval_name, dataloader in val_dataloaders.items():
            for model_type in self.eval_models:
                if model_type == "ema":
                    ema = self.ema_callback.ema
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())

                    # Compute ema/eval/eval_loss
                    ema_eval_loss = None
                    if trainer.is_global_zero:
                        ema_losses = []
                        model.eval()
                        with torch.no_grad():
                            for batch in dataloader:
                                batch = trainer.strategy.batch_to_device(batch, device)
                                loss = model.architecture(batch)
                                ema_losses.append(loss.item() if hasattr(loss, "item") else float(loss))
                        
                        model.train()
                        
                        ema_eval_loss = float(np.mean(ema_losses))

                prefix = f"{model_type}/{eval_name}"
                
                if trainer.is_global_zero:
                    if train_loss is not None:
                        all_metrics[f"{model_type}/{eval_name}/train_loss"] = train_loss
                    
                    eval_loss = std_eval_loss if model_type == "standard" else (ema_eval_loss if model_type == "ema" else None)
                    if eval_loss is not None:
                        all_metrics[f"{model_type}/{eval_name}/eval_loss"] = eval_loss

                metrics = evaluate(
                    model,
                    dataloader,
                    device,
                    self.metrics_args,
                    prefix=prefix,
                    log=False,
                    rank=getattr(trainer, "global_rank", None),
                    world_size=getattr(trainer, "world_size", None),
                )
                all_metrics.update(metrics)

                if model_type == "ema":
                    ema.restore(model.parameters())

        if trainer.is_global_zero:
            trainer.logger.log_metrics(all_metrics, step=trainer.global_step)


def compute_metrics(
    pred_df: pd.DataFrame,
    target_df: pd.DataFrame,
    metrics_args: dict[str, Any] = None,
) -> dict[str, float]:
    """Compute pred vs. target metrics per (CELL_TYPE, EXPERIMENT, PERTURBATIONS) and then average across subpopulations."""
    if len(pred_df) == 0:
        raise ValueError("No predictions or targets to compute metrics.")

    # Make PERTURBATIONS hashable for grouping (list/dict -> stable string)
    def _hashable(row: pd.Series) -> tuple:
        cell, exp, p = row.get(CELL_TYPE), row.get(EXPERIMENT), row.get(PERTURBATIONS)
        if isinstance(p, (list, dict)):
            p = json.dumps(p, sort_keys=True)
        
        return (cell, exp, p)

    pred_df = pred_df.copy()
    target_df = target_df.copy()

    pred_df["_subpop"] = pred_df.apply(_hashable, axis=1)
    target_df["_subpop"] = target_df.apply(_hashable, axis=1)
    
    subpop_values = pred_df["_subpop"].unique()
    
    metrics_dict = defaultdict(lambda: defaultdict(list))

    for pop_id in subpop_values:
        pop_mask = pred_df["_subpop"] == pop_id

        pred_expr = np.stack(pred_df.loc[pop_mask, "predicted_expression"].values)
        target_expr = np.stack(target_df.loc[pop_mask, "target_expression"].values)

        for metric_level, metric_names in metrics_args.items():
            for metric_name in metric_names:
                metrics_dict[metric_level][metric_name].append(METRIC_FUNCTION_DICT[metric_level][metric_name](pred_expr, target_expr))

    return {
        f"{level}_{name}": float(np.nanmean(vals))
        for level, names_dict in metrics_dict.items()
        for name, vals in names_dict.items()
    }


def evaluate(
    model: pl.LightningModule,
    dataloader: DataLoader,
    device: torch.device | str,
    metrics_args: dict[str, Any],
    prefix: str = "eval",
    log: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
) -> dict[str, float]:
    """
    Two-step process folllowed by memory cleanup and wandb logging:
        1. Run inference
        2. Compute metrics
    """
    # Run inference
    pred_df, target_df = run_inference(model, dataloader, device)
    
    # Compute metrics
    metrics = compute_metrics(pred_df, target_df, metrics_args=metrics_args)

    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    # Memory cleanup
    del pred_df, target_df
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()

    # Log metrics to wandb
    if log:
        wandb.log({k: v for k, v in metrics.items()})

    return metrics
