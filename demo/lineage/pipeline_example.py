#!/usr/bin/env python
"""Example pipeline with W&B lineage + DVC storage.

Shows how artifacts flow through pipeline stages:
1. Pretrain: consumes data → produces checkpoint + embeddings
2. Finetune: consumes checkpoint + embeddings → produces checkpoint
3. Eval: consumes checkpoint → produces metrics

Run:
    python demo/lineage/pipeline_example.py
"""

from pathlib import Path

import wandb

from demo.lineage.artifacts import log_checkpoint, log_dvc_artifact, use_dvc_artifact


def pretrain(data_artifact: str, output_dir: Path) -> tuple[str, str]:
    """Pretraining stage.

    Consumes: data (DVC)
    Produces: checkpoint (W&B) + embeddings (DVC)
    """
    with wandb.init(project="hooke", job_type="pretrain") as run:
        # Consume data — creates lineage edge
        data_path = use_dvc_artifact(data_artifact)
        print(f"Training on: {data_path}")

        # Mock training
        checkpoint_path = output_dir / "pretrain_checkpoint.pt"
        embeddings_path = output_dir / "embeddings.zarr"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("mock checkpoint")
        embeddings_path.mkdir(parents=True, exist_ok=True)

        # Log checkpoint to W&B (small file, direct upload)
        log_checkpoint(
            "pretrain-checkpoint",
            checkpoint_path,
            metadata={"step": 200000, "loss": 0.1},
        )

        # Log embeddings to DVC (large file, reference only)
        # First: dvc add outputs/embeddings.zarr && dvc push
        log_dvc_artifact(
            "pretrain-embeddings",
            str(embeddings_path) + ".dvc",
            artifact_type="embeddings",
            metadata={"shape": [100000, 512]},
        )

        return f"pretrain-checkpoint:v{run.id}", f"pretrain-embeddings:v{run.id}"


def finetune(
    checkpoint_artifact: str, embeddings_artifact: str, output_dir: Path
) -> str:
    """Finetuning stage.

    Consumes: checkpoint (W&B) + embeddings (DVC)
    Produces: checkpoint (W&B)
    """
    with wandb.init(project="hooke", job_type="finetune") as run:
        # Consume artifacts — W&B tracks lineage automatically
        checkpoint = wandb.use_artifact(checkpoint_artifact)
        checkpoint_dir = checkpoint.download()
        print(f"Loaded checkpoint: {checkpoint_dir}")

        embeddings_path = use_dvc_artifact(embeddings_artifact)
        print(f"Loaded embeddings: {embeddings_path}")

        # Mock finetuning
        ft_checkpoint_path = output_dir / "finetune_checkpoint.pt"
        ft_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ft_checkpoint_path.write_text("mock finetuned checkpoint")

        # Log finetuned checkpoint
        log_checkpoint(
            "finetune-checkpoint",
            ft_checkpoint_path,
            metadata={"base_checkpoint": checkpoint_artifact, "epochs": 20},
        )

        return f"finetune-checkpoint:v{run.id}"


def evaluate(checkpoint_artifact: str) -> dict:
    """Evaluation stage.

    Consumes: checkpoint (W&B)
    Produces: metrics (logged to W&B)
    """
    with wandb.init(project="hooke", job_type="eval") as run:
        # Consume checkpoint
        checkpoint = wandb.use_artifact(checkpoint_artifact)
        checkpoint_dir = checkpoint.download()
        print(f"Evaluating: {checkpoint_dir}")

        # Mock evaluation
        metrics = {
            "map_cosine": 0.85,
            "pearson_delta": 0.72,
            "pathway_capture": 0.68,
        }

        # Log metrics
        wandb.log(metrics)
        wandb.summary.update(metrics)

        return metrics


def main():
    """Run full pipeline."""
    output_dir = Path("outputs/demo")

    print("=" * 60)
    print("Stage 1: Pretrain")
    print("=" * 60)
    # Assumes: dvc add data/pretraining.parquet && dvc push
    ckpt_art, emb_art = pretrain("pretraining-v6:latest", output_dir / "pretrain")

    print()
    print("=" * 60)
    print("Stage 2: Finetune")
    print("=" * 60)
    ft_ckpt_art = finetune(ckpt_art, emb_art, output_dir / "finetune")

    print()
    print("=" * 60)
    print("Stage 3: Evaluate")
    print("=" * 60)
    metrics = evaluate(ft_ckpt_art)

    print()
    print("=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Final metrics: {metrics}")
    print()
    print("View lineage graph at: https://wandb.ai/<entity>/hooke/artifacts")


if __name__ == "__main__":
    main()
