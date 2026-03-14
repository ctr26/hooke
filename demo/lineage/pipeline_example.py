#!/usr/bin/env python
"""Pipeline with W&B artifacts (storage + lineage).

W&B handles everything — no external tools.

Run:
    python demo/lineage/pipeline_example.py
"""

from pathlib import Path

import wandb

from demo.lineage.artifacts import log_artifact, log_reference, use_artifact


def pretrain(data_artifact: str, output_dir: Path) -> str:
    """Pretraining stage."""
    with wandb.init(project="hooke", job_type="pretrain") as run:
        # Consume data artifact
        data_path = use_artifact(data_artifact)
        print(f"Training on: {data_path}")

        # Mock training
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("mock checkpoint")

        # Log checkpoint to W&B
        log_artifact(
            "pretrain-checkpoint",
            checkpoint_path,
            artifact_type="model",
            metadata={"step": 200000, "loss": 0.1},
        )

        return f"pretrain-checkpoint:v{run.id}"


def finetune(checkpoint_artifact: str, output_dir: Path) -> str:
    """Finetuning stage."""
    with wandb.init(project="hooke", job_type="finetune") as run:
        # Consume checkpoint — lineage tracked automatically
        ckpt_path = use_artifact(checkpoint_artifact)
        print(f"Loaded checkpoint: {ckpt_path}")

        # Mock finetuning
        ft_path = output_dir / "checkpoint.pt"
        ft_path.parent.mkdir(parents=True, exist_ok=True)
        ft_path.write_text("mock finetuned")

        log_artifact(
            "finetune-checkpoint",
            ft_path,
            artifact_type="model",
            metadata={"base": checkpoint_artifact, "epochs": 20},
        )

        return f"finetune-checkpoint:v{run.id}"


def evaluate(checkpoint_artifact: str) -> dict:
    """Evaluation stage."""
    with wandb.init(project="hooke", job_type="eval") as run:
        ckpt_path = use_artifact(checkpoint_artifact)
        print(f"Evaluating: {ckpt_path}")

        metrics = {"map_cosine": 0.85, "pearson": 0.72}
        wandb.log(metrics)

        return metrics


def main():
    output = Path("outputs/demo")

    # For large data on shared FS: use reference (no upload)
    # log_reference("pretraining-v6", "file:///rxrx/data/pretraining.parquet")

    ckpt = pretrain("pretraining-v6:latest", output / "pretrain")
    ft_ckpt = finetune(ckpt, output / "finetune")
    metrics = evaluate(ft_ckpt)

    print(f"Done: {metrics}")


if __name__ == "__main__":
    main()
