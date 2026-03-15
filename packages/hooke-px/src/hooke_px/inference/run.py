"""Inference execution — bridges hooke_px pipeline to hooke_forge distributed inference."""

import logging
from pathlib import Path

from hooke_forge.inference.checkpoint import extract_model_config, find_checkpoint

log = logging.getLogger(__name__)


def resolve_checkpoint(checkpoint_path: str) -> Path:
    """Resolve checkpoint from various formats.

    Supports:
    - Local file path: /path/to/step_200000.ckpt
    - Training dir + step: /path/to/training_dir (finds latest checkpoint)
    - W&B artifact: downloaded artifact directory
    """
    path = Path(checkpoint_path)

    # Direct checkpoint file
    if path.is_file() and path.suffix == ".ckpt":
        return path

    # Training directory — find the latest checkpoint
    if path.is_dir():
        checkpoints_dir = path / "checkpoints"
        if checkpoints_dir.exists():
            ckpts = sorted(checkpoints_dir.glob("step_*.ckpt"))
            if ckpts:
                return ckpts[-1]

        # Try nested structure (timestamp/job_id/checkpoints/)
        for subdir in path.iterdir():
            if subdir.is_dir():
                nested_ckpts = subdir / "checkpoints"
                if nested_ckpts.exists():
                    ckpts = sorted(nested_ckpts.glob("step_*.ckpt"))
                    if ckpts:
                        return ckpts[-1]

    raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")


def run_inference_job(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    num_samples: int,
) -> str:
    """Run distributed inference via hooke_forge.

    Bridges the hooke_px pipeline schema to the existing
    hooke_forge distributed inference infrastructure.

    Args:
        checkpoint_path: Path to model checkpoint or training directory
        dataset_path: Path to input metadata parquet
        output_dir: Output directory for features
        batch_size: Batch size per worker GPU
        num_workers: Number of SLURM workers
        num_samples: Samples per observation

    Returns:
        Path to output features directory (contains {representation}.zarr files)
    """
    from hooke_forge.inference.distributed import run_distributed_inference

    checkpoint = resolve_checkpoint(checkpoint_path)
    log.info(f"Resolved checkpoint: {checkpoint}")

    model_config = extract_model_config(checkpoint.parent.parent)
    if model_config:
        log.info(f"Extracted model config: {model_config}")

    result_dir = run_distributed_inference(
        checkpoint_path=checkpoint,
        input_parquet=Path(dataset_path),
        output_dir=Path(output_dir),
        model_config=model_config or None,
        num_workers=num_workers,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    features_path = str(result_dir / "features")
    log.info(f"Inference complete: {features_path}")
    return features_path
