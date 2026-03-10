"""Evaluation: load checkpoint, compute metrics on a validation set.

Mirrors hooke-forge's eval.py pattern:
- Load model from checkpoint
- Run on validation data
- Compute and report metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from hsh.config import EvalConfig
from hsh.data import make_dataloaders
from hsh.train import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def evaluate(config: EvalConfig) -> dict:
    """Evaluate a checkpoint on synthetic validation data.

    Returns:
        Dict with metrics (mse_loss, mean_pred_norm).
    """
    model, state = load_checkpoint(config.checkpoint)
    step = state["step"]
    log.info("Evaluating checkpoint from step %d: %s", step, config.checkpoint)

    _, val_loader = make_dataloaders(
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        input_dim=model.config.input_dim,
    )

    model.eval()
    losses = []
    pred_norms = []

    with torch.no_grad():
        for batch in val_loader:
            loss = model.loss(batch["x0"], batch["x1"])
            losses.append(loss.item())

            preds = model.generate(batch["x0"], num_steps=5)
            pred_norms.append(preds.norm(dim=-1).mean().item())

    metrics = {
        "step": step,
        "mse_loss": sum(losses) / len(losses),
        "mean_pred_norm": sum(pred_norms) / len(pred_norms),
    }

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics: %s", metrics)
    log.info("Saved to %s", metrics_path)

    return metrics


def cli() -> None:
    """CLI entry point for hsh-eval."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="./eval_outputs")
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
    evaluate(config)


if __name__ == "__main__":
    cli()
