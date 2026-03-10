"""Finetune from a pretrained checkpoint.

Thin wrapper around ``train()`` that loads a pretrained checkpoint
and continues training with (typically lower) learning rate.
"""

from __future__ import annotations

import argparse
import logging

from hsh.config import FinetuneConfig
from hsh.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def finetune(config: FinetuneConfig) -> dict:
    """Finetune a model by resuming training from a checkpoint.

    Returns:
        Dict with final step, loss, and checkpoint path.
    """
    return train(config.train, resume_from=config.base_checkpoint)


def cli() -> None:
    """CLI entry point for hsh-finetune."""
    parser = argparse.ArgumentParser(description="Finetune from a pretrained checkpoint")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="./finetune_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = FinetuneConfig(
        base_checkpoint=args.base_checkpoint,
        train=dict(
            num_steps=args.num_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        ),
    )
    result = finetune(config)
    log.info("Finetuning complete: %s", result)


if __name__ == "__main__":
    cli()
