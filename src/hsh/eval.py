"""Evaluation: load checkpoint, compute metrics on a validation set.

Mirrors hooke-forge's eval.py pattern:
- Load model from checkpoint
- Run on validation data
- Compute and report metrics
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from hydra_zen import store, zen

from hsh.config import EvalConfig
from hsh.data import make_dataloaders
from hsh.hydra_conf import EvalTaskConf
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
    """CLI entry point for hsh-eval.

    Override fields with Hydra's ``key=value`` syntax, e.g.::

        hsh-eval config.checkpoint=path/to/ckpt config.batch_size=16
    """
    store(EvalTaskConf, name="hsh_eval")
    store.add_to_hydra_store()
    zen(evaluate).hydra_main(config_name="hsh_eval", version_base="1.3", config_path=None)


if __name__ == "__main__":
    cli()
