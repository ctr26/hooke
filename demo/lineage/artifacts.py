"""Artifact tracking with W&B (storage + lineage).

W&B handles both storage and lineage - no external tools.
"""

from pathlib import Path

import wandb


def log_artifact(
    name: str,
    path: Path,
    artifact_type: str = "dataset",
    metadata: dict | None = None,
) -> wandb.Artifact:
    """Log artifact to W&B (uploaded and tracked).

    Args:
        name: Artifact name
        path: Path to file or directory
        artifact_type: Type (dataset, model, embeddings, etc.)
        metadata: Additional metadata

    Returns:
        Logged W&B Artifact
    """
    artifact = wandb.Artifact(name, type=artifact_type)

    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    if metadata:
        artifact.metadata.update(metadata)

    wandb.log_artifact(artifact)
    return artifact


def use_artifact(artifact_name: str) -> Path:
    """Download artifact from W&B.

    Args:
        artifact_name: W&B artifact (e.g., "pretraining-v6:latest")

    Returns:
        Local path to downloaded artifact
    """
    artifact = wandb.use_artifact(artifact_name)
    return Path(artifact.download())


def log_reference(
    name: str,
    uri: str,
    artifact_type: str = "dataset",
    metadata: dict | None = None,
) -> wandb.Artifact:
    """Log reference to external data (no upload).

    For data already on shared FS or S3.

    Args:
        name: Artifact name
        uri: URI to data (file://, s3://, gs://)
        artifact_type: Type
        metadata: Additional metadata

    Returns:
        Logged W&B Artifact
    """
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_reference(uri)

    if metadata:
        artifact.metadata.update(metadata)

    wandb.log_artifact(artifact)
    return artifact
