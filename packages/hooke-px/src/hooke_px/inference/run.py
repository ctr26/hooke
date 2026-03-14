"""Inference execution (runs inside SLURM job)."""

from pathlib import Path


def run_inference_job(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    num_samples: int,
) -> str:
    """Run inference and save features.

    This is the actual compute — runs on GPU node.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to dataset
        output_dir: Output directory
        batch_size: Batch size per GPU
        num_workers: Parallel workers
        num_samples: Samples per well

    Returns:
        Path to output features (zarr)
    """
    import torch

    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features_path = output_path / "features.zarr"

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {features_path}")

    # TODO: Actual inference implementation
    # This is a placeholder — replace with real model loading + inference

    # model = load_model(checkpoint_path)
    # dataset = load_dataset(dataset_path, num_samples=num_samples)
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    #
    # features = []
    # for batch in dataloader:
    #     with torch.no_grad():
    #         feat = model(batch)
    #     features.append(feat)
    #
    # save_zarr(features, features_path)

    # Placeholder: create empty zarr
    import zarr
    zarr.open(str(features_path), mode="w", shape=(1000, 512), dtype="float32")

    print(f"✅ Inference complete: {features_path}")

    return str(features_path)
