"""Training loop and CLI entry point.

Mirrors hooke-forge's trainer.py pattern:
- Step-based loop (not epoch-based)
- Periodic checkpoint saving
- Periodic evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch

from hsh.config import ModelConfig, TrainConfig
from hsh.data import make_dataloaders
from hsh.model import FlowMatchingMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def save_checkpoint(
    model: FlowMatchingMLP,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: ModelConfig,
    output_dir: str,
) -> Path:
    """Save a training checkpoint."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": model_config.model_dump(),
        },
        path,
    )
    log.info("Saved checkpoint: %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[FlowMatchingMLP, dict]:
    """Load model and metadata from a checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**state["model_config"])
    model = FlowMatchingMLP(config)
    model.load_state_dict(state["model"])
    return model, state


def train(
    train_config: TrainConfig,
    model_config: ModelConfig | None = None,
) -> dict:
    """Run the training loop.

    Returns:
        Dict with final step, loss, and checkpoint path.
    """
    if model_config is None:
        model_config = ModelConfig()

    torch.manual_seed(train_config.seed)

    model = FlowMatchingMLP(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    train_loader, val_loader = make_dataloaders(
        batch_size=train_config.batch_size,
        num_samples=train_config.num_samples,
        input_dim=model_config.input_dim,
        seed=train_config.seed,
    )

    os.makedirs(train_config.output_dir, exist_ok=True)

    train_iter = iter(train_loader)
    last_ckpt_path = None
    last_loss = float("nan")

    for step in range(1, train_config.num_steps + 1):
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

        if step % max(train_config.num_steps // 10, 1) == 0:
            log.info("step=%d  loss=%.6f", step, last_loss)

        if step % train_config.ckpt_every == 0 or step == train_config.num_steps:
            last_ckpt_path = save_checkpoint(model, optimizer, step, model_config, train_config.output_dir)

        if step % train_config.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_losses.append(model.loss(val_batch["x0"], val_batch["x1"]).item())
            val_loss = sum(val_losses) / len(val_losses)
            log.info("step=%d  val_loss=%.6f", step, val_loss)

    config_path = Path(train_config.output_dir) / "config.json"
    config_path.write_text(
        json.dumps({"train": train_config.model_dump(), "model": model_config.model_dump()}, indent=2)
    )

    return {"step": step, "loss": last_loss, "checkpoint": str(last_ckpt_path)}


def cli() -> None:
    """CLI entry point for hsh-train."""
    parser = argparse.ArgumentParser(description="Train a minimal flow-matching model")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=32)
    args = parser.parse_args()

    model_config = ModelConfig(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    train_config = TrainConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    result = train(train_config, model_config)
    log.info("Training complete: %s", result)


if __name__ == "__main__":
    cli()
