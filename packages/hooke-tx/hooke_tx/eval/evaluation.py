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
    "aggregated": {
        "mae": lambda x, y: np.mean(np.abs(x.mean(axis=0) - y.mean(axis=0))),
        "mse": lambda x, y: np.mean((x.mean(0) - y.mean(0)) ** 2),
        "pearson": lambda x, y: pearsonr(x.mean(0), y.mean(0))[0],
    },
    "distributed": {
        "mmd": lambda pred_expr, target_expr: compute_e_distance(pred_expr, target_expr),
    },
}


class EvalCallback(Callback):
    """Runs inference + evaluation metrics at the end of each validation epoch and logs them."""
    def __init__(self, metrics_config: dict[str, Any]) -> None:
        super().__init__()
        
        self.metrics_config = metrics_config

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
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
        
        for eval_name, dataloader in val_dataloaders.items():
            metrics = evaluate(
                model,
                dataloader,
                device,
                self.metrics_config,
                prefix=eval_name,
                log=False,
                rank=getattr(trainer, "global_rank", None),
                world_size=getattr(trainer, "world_size", None),
            )
            
            if trainer.is_global_zero:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)


def compute_metrics(
    pred_df: pd.DataFrame,
    target_df: pd.DataFrame,
    metrics_config: dict[str, Any] = None,
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

        for metric_level, metric_names in metrics_config.items():
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
    metrics_config: dict[str, Any],
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
    metrics = compute_metrics(pred_df, target_df, metrics_config=metrics_config)

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
