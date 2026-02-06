#!/usr/bin/env python3
"""
Prepare model predictions for VCB evaluation.

This script creates the directory structure needed for VCB evaluation by:
1. Creating symlinks to prediction files with VCB-expected naming
2. Setting biological_context based on the task (virtual_map or phenorescue)
3. Generating split.json from the split column in prepared_metadata.parquet

Usage:
    uv run python scripts/prepare_predictions_for_evaluation.py \
        --predictions-dir /path/to/step_200000 \
        --ground-truth-dir /path/to/drugscreen__cell_paint__v1_2 \
        --output-dir /path/to/step_200000_eval \
        --zarr-feature-name pred_phenom
        --task-id phenorescue

Tasks:
    virtual_map  - biological_context: ["cell_type"]
    phenorescue  - biological_context: ["plate_disease_model", "cell_type"]
"""

import argparse
import json
import os
from pathlib import Path

import polars as pl


TASK_BIOLOGICAL_CONTEXT = {
    "virtual_map": ["cell_type"],
    "phenorescue": ["plate_disease_model", "cell_type"],
}


def get_biological_context(task_id: str) -> list[str]:
    """Get biological_context for the specified task."""
    if task_id not in TASK_BIOLOGICAL_CONTEXT:
        raise ValueError(
            f"Unknown task_id: {task_id}. Must be one of: {list(TASK_BIOLOGICAL_CONTEXT.keys())}"
        )
    return TASK_BIOLOGICAL_CONTEXT[task_id]


def create_predictions_directory(
    predictions_dir: Path,
    output_dir: Path,
    task_id: str,
    zarr_feat_name: str,
) -> Path:
    """Create predictions directory with symlinks and metadata files."""
    print(f"Creating predictions directory in {output_dir}")

    pred_output = output_dir / "predictions"
    pred_output.mkdir(parents=True, exist_ok=True)

    # Source files
    metadata_file = predictions_dir / "prepared_metadata.parquet"
    features_file = predictions_dir / "features" / f"{zarr_feat_name}.zarr"

    # Verify source files exist
    for src in [metadata_file, features_file]:
        if not src.exists():
            raise FileNotFoundError(f"Required source file not found: {src}")

    # Create symlinks
    obs_link = pred_output / "obs.parquet"
    features_link = pred_output / "features.zarr"

    for link in [obs_link, features_link]:
        if link.is_symlink() or link.exists():
            link.unlink()

    obs_link.symlink_to(os.path.relpath(metadata_file, pred_output))
    features_link.symlink_to(os.path.relpath(features_file, pred_output))
    print(f"  Created symlinks: obs.parquet, features.zarr")

    # Create dataset metadata with task-appropriate biological_context
    biological_context = get_biological_context(task_id)
    pred_metadata = {
        "dataset_id": "predictions",
        "description": "Predicted phenomics features",
        "version": "1.0",
        "biological_context": biological_context,
    }

    with open(pred_output / "predictions_dataset_metadata.json", "w") as f:
        json.dump(pred_metadata, f, indent=2)
    print(
        f"  Created predictions_dataset_metadata.json with biological_context: {biological_context}"
    )

    # Create var file (1664 phenom features)
    var_df = pl.DataFrame({"feature_name": [f"feature_{i}" for i in range(1664)]})
    var_df.write_parquet(pred_output / "predictions_var.parquet")
    print(f"  Created predictions_var.parquet with 1664 features")

    return pred_output


def create_split_json(
    predictions_dir: Path,
    output_dir: Path,
) -> Path:
    """Create split.json from the split column in prepared_metadata.parquet."""
    print("Creating split.json from metadata split column...")

    metadata_file = predictions_dir / "prepared_metadata.parquet"
    df = pl.read_parquet(metadata_file)

    # Get split column values
    split_values = df["split"].unique().to_list()
    print(f"  Found split values: {split_values}")

    # Identify split categories
    train_splits = [s for s in split_values if s == "train"]
    valid_splits = [s for s in split_values if s.startswith("valid")]
    test_splits = [s for s in split_values if s.startswith("test")]

    # Map validation perturbations to finetune (for evaluation reference)
    # and test perturbations to test
    finetune_indices = df.filter(pl.col("split").is_in(valid_splits))[
        "zarr_index"
    ].to_list()

    test_indices = df.filter(pl.col("split").is_in(test_splits))["zarr_index"].to_list()

    # Get control indices (separate from splits)
    control_indices = df.filter(pl.col("is_negative_control"))["zarr_index"].to_list()

    # Get base_state indices if column exists (separate from splits)
    base_state_indices = []
    if "is_base_state" in df.columns:
        base_state_indices = df.filter(pl.col("is_base_state"))["zarr_index"].to_list()

    # Build split structure
    split_data = {
        "dataset_id": "predictions",
        "version": 1,
        "splitting_level": "compound",
        "splitting_strategy": "random",
        "folds": [
            {
                "outer_fold": 0,
                "inner_fold": 0,
                "finetune": finetune_indices,
                "test": test_indices,
            }
        ],
        "controls": control_indices,
        "base_states": base_state_indices,
    }

    split_file = output_dir / "split.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    print(f"  Split summary:")
    print(
        f"    finetune: {len(finetune_indices)} observations (using validation perturbations)"
    )
    print(f"    test: {len(test_indices)} observations (test perturbations)")
    print(f"    controls: {len(control_indices)} observations")
    print(f"    base_states: {len(base_state_indices)} observations")
    print(f"  Created {split_file}")

    return split_file


def print_next_steps(output_dir: Path, ground_truth_dir: Path, task_id: str):
    """Print the VCB command to run evaluation."""
    pred_path = output_dir / "predictions"
    split_path = output_dir / "split.json"
    results_path = output_dir / "results"

    print("\nNext steps - run VCB evaluation:")
    print(f"""
cd /rxrx/data/user/jason.hartford/claude/vcb
uv run vcb evaluate predictions px \\
    -p {pred_path} \\
    -t {ground_truth_dir} \\
    -s {split_path} \\
    -o {results_path} \\
    --task-id {task_id} \\
    --predictions-zarr-index-column zarr_index \\
    --no-distributional-metrics
""")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare model predictions for VCB evaluation"
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing prepared_metadata.parquet and features/pred_phenom.zarr",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Ground truth dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation setup",
    )
    parser.add_argument(
        "--zarr-feature-name",
        type=str,
        default="pred_phenom",
        help="Name of the zarr feature file to use (default: pred_phenom)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        choices=list(TASK_BIOLOGICAL_CONTEXT.keys()),
        help="VCB task to prepare for (determines biological_context)",
    )

    args = parser.parse_args()

    print(f"Preparing predictions for VCB evaluation")
    print(f"  Predictions: {args.predictions_dir}")
    print(f"  Ground truth: {args.ground_truth_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Task: {args.task_id}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create predictions directory with symlinks and metadata
    create_predictions_directory(
        args.predictions_dir, args.output_dir, args.task_id, args.zarr_feature_name
    )

    # Create split.json
    create_split_json(args.predictions_dir, args.output_dir)

    # Print next steps
    print_next_steps(args.output_dir, args.ground_truth_dir, args.task_id)

    print("Done!")


if __name__ == "__main__":
    main()
