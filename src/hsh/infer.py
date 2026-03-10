"""Inference: load checkpoint, generate predictions, write outputs.

Mirrors hooke-forge's inference pattern:
- Load model from checkpoint
- Run forward pass on input data
- Save predictions to output directory
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from hsh.config import InferConfig
from hsh.data import SyntheticDataset, collate_fn
from hsh.train import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def infer(config: InferConfig) -> dict:
    """Run inference and save predictions.

    Returns:
        Dict with output path and prediction shape.
    """
    model, state = load_checkpoint(config.checkpoint)
    step = state["step"]
    log.info("Running inference with checkpoint from step %d: %s", step, config.checkpoint)

    dataset = SyntheticDataset(
        num_samples=config.num_samples,
        input_dim=model.config.input_dim,
        seed=99,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            preds = model.generate(batch["x0"], num_steps=10)
            all_predictions.append(preds)

    predictions = torch.cat(all_predictions, dim=0)

    os.makedirs(config.output_dir, exist_ok=True)
    output_path = Path(config.output_dir) / f"predictions_step_{step}.pt"
    torch.save(predictions, output_path)
    log.info("Saved %s predictions to %s", list(predictions.shape), output_path)

    return {"output_path": str(output_path), "shape": list(predictions.shape)}


def cli() -> None:
    """CLI entry point for hsh-infer."""
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./predictions")
    args = parser.parse_args()

    config = InferConfig(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
    infer(config)


if __name__ == "__main__":
    cli()
