"""Finetune from a pretrained checkpoint.

Mirrors hooke-forge's finetuning pattern:
- Load model + optimizer from checkpoint
- Optionally override learning rate
- Continue training for N more steps
- Save new checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from hsh.config import FinetuneConfig
from hsh.data import make_dataloaders
from hsh.train import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def finetune(config: FinetuneConfig) -> dict:
    """Finetune a model from a pretrained checkpoint.

    Returns:
        Dict with final step, loss, and checkpoint path.
    """
    torch.manual_seed(config.seed)

    model, state = load_checkpoint(config.base_checkpoint)
    model_config = model.config
    start_step = state["step"]
    log.info("Loaded checkpoint from step %d: %s", start_step, config.base_checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader, _ = make_dataloaders(
        batch_size=config.batch_size,
        num_samples=config.num_samples,
        input_dim=model_config.input_dim,
        seed=config.seed,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    train_iter = iter(train_loader)
    last_loss = float("nan")
    last_ckpt_path = None

    for step in range(start_step + 1, start_step + config.num_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = model.loss(batch["x0"], batch["x1"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

        if step % max(config.num_steps // 5, 1) == 0:
            log.info("finetune step=%d  loss=%.6f", step, last_loss)

    last_ckpt_path = save_checkpoint(model, optimizer, step, model_config, config.output_dir)

    return {"step": step, "loss": last_loss, "checkpoint": str(last_ckpt_path)}


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
        num_steps=args.num_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    result = finetune(config)
    log.info("Finetuning complete: %s", result)


if __name__ == "__main__":
    cli()
