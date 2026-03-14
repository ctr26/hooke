"""Artifact tracking: W&B lineage + DVC storage.

Pattern:
- DVC stores large files (data, embeddings) on shared FS
- W&B tracks lineage (run → artifact → run)
- Code uses W&B API, which resolves to DVC paths
"""

import subprocess
from pathlib import Path

import wandb


def log_dvc_artifact(
    name: str,
    dvc_path: str,
    artifact_type: str = "dataset",
    metadata: dict | None = None,
) -> wandb.Artifact:
    """Log a DVC-tracked file as W&B artifact (reference only).

    Args:
        name: Artifact name (e.g., "pretraining-v6")
        dvc_path: Path to .dvc file or tracked directory
        artifact_type: W&B artifact type (dataset, embeddings, etc.)
        metadata: Additional metadata to attach

    Returns:
        Logged W&B Artifact
    """
    artifact = wandb.Artifact(name, type=artifact_type)

    # Store DVC reference
    artifact.metadata["dvc_path"] = dvc_path
    artifact.metadata["dvc_tracked"] = True

    # Get DVC remote info if available
    try:
        result = subprocess.run(
            ["dvc", "remote", "default"], capture_output=True, text=True, check=True
        )
        artifact.metadata["dvc_remote"] = result.stdout.strip()
    except subprocess.CalledProcessError:
        artifact.metadata["dvc_remote"] = "unknown"

    # Add custom metadata
    if metadata:
        artifact.metadata.update(metadata)

    # Log artifact (no data uploaded — just metadata)
    wandb.log_artifact(artifact)
    return artifact


def use_dvc_artifact(artifact_name: str, pull: bool = True) -> Path:
    """Get local path to DVC data via W&B artifact.

    Args:
        artifact_name: W&B artifact name (e.g., "pretraining-v6:latest")
        pull: Whether to run dvc pull

    Returns:
        Local path to data
    """
    artifact = wandb.use_artifact(artifact_name)

    if not artifact.metadata.get("dvc_tracked"):
        raise ValueError(f"Artifact {artifact_name} is not DVC-tracked")

    dvc_path = artifact.metadata["dvc_path"]

    if pull:
        subprocess.run(["dvc", "pull", dvc_path], check=True)

    # Return path without .dvc suffix
    path = Path(dvc_path)
    if path.suffix == ".dvc":
        return path.with_suffix("")
    return path


def log_checkpoint(
    name: str,
    checkpoint_path: Path,
    artifact_type: str = "model",
    metadata: dict | None = None,
) -> wandb.Artifact:
    """Log a model checkpoint to W&B (uploaded).

    For small-ish files (<5GB), upload directly to W&B.

    Args:
        name: Artifact name
        checkpoint_path: Path to checkpoint file
        artifact_type: W&B artifact type
        metadata: Additional metadata

    Returns:
        Logged W&B Artifact
    """
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(str(checkpoint_path))

    if metadata:
        artifact.metadata.update(metadata)

    wandb.log_artifact(artifact)
    return artifact
