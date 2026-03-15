#!/usr/bin/env python
"""Upload pretrained checkpoint to W&B for Weave lineage.

Usage:
    python scripts/upload_checkpoint.py /path/to/checkpoint.pt --name pretrain-checkpoint

This creates a W&B artifact that Weave can reference:
    weave.ref("hooke-px/pretrain-checkpoint:latest").get()
"""

import argparse
from pathlib import Path

import wandb


def upload_checkpoint(
    checkpoint_path: Path,
    name: str = "pretrain-checkpoint",
    project: str = "hooke-px",
    metadata: dict | None = None,
) -> str:
    """Upload checkpoint to W&B as artifact and publish to Weave.

    Creates both a W&B artifact (for storage) and a Weave object (for lineage).
    The inference step can resolve the checkpoint via either mechanism.

    Args:
        checkpoint_path: Path to checkpoint file
        name: Artifact name
        project: W&B project name
        metadata: Additional metadata (step, config, etc.)

    Returns:
        Artifact reference string
    """
    run = wandb.init(project=project, job_type="upload")

    artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=metadata or {},
    )

    # Add checkpoint file
    artifact.add_file(str(checkpoint_path))

    # Log artifact
    run.log_artifact(artifact)
    run.finish()

    # Also publish to Weave for lineage tracking
    try:
        import weave

        weave.init(project)
        weave.publish(
            {
                "path": str(checkpoint_path),
                "artifact_ref": f"{project}/{name}:latest",
                "metadata": metadata or {},
            },
            name=name,
        )
        print(f"   Weave: published as {project}/{name}")
    except Exception as e:
        print(f"   Weave publish skipped: {e}")

    ref = f"{project}/{name}:latest"
    print(f"Uploaded: {ref}")
    print(f"   View: https://wandb.ai/valencelabs/{project}/artifacts/model/{name}")

    return ref


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to W&B")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--name", default="pretrain-checkpoint", help="Artifact name")
    parser.add_argument("--project", default="hooke-px", help="W&B project")
    parser.add_argument("--step", type=int, help="Training step")
    parser.add_argument("--config", type=Path, help="Path to config JSON")

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    metadata = {}
    if args.step:
        metadata["step"] = args.step
    if args.config and args.config.exists():
        import json
        metadata["config"] = json.loads(args.config.read_text())

    upload_checkpoint(
        args.checkpoint,
        name=args.name,
        project=args.project,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
