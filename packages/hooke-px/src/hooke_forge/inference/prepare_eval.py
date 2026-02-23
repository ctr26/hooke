"""Prepare predictions for VCB evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import polars as pl

log = logging.getLogger(__name__)

TASK_BIOLOGICAL_CONTEXT = {
    "virtual_map": ["cell_type"],
    "phenorescue": ["plate_disease_model", "cell_type"],
}


def prepare_for_vcb(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_dir: Path,
    task_id: str,
    lineage: Optional[dict] = None,
    test_only: bool = True,
) -> Path:
    """Prepare predictions directory for VCB evaluation.

    Creates:
    - predictions/ (default, test-only) or predictions_with_valid/ subdirectory
    - split.json or split_with_valid.json with corrected ground truth indices

    Args:
        predictions_dir: Directory with prepared_metadata.parquet and features/
        ground_truth_dir: VCB ground truth dataset directory
        output_dir: Where to create evaluation structure
        task_id: VCB task (virtual_map or phenorescue)
        lineage: Optional model lineage dict from get_model_lineage()
        test_only: If True (default), filter to test observations only to match VCB expectations

    Returns:
        Path to the created eval directory
    """
    predictions_dir = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = Path(output_dir)

    if task_id not in TASK_BIOLOGICAL_CONTEXT:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Must be one of: {list(TASK_BIOLOGICAL_CONTEXT.keys())}"
        )

    # Determine output naming based on test_only parameter
    if test_only:
        pred_dir_name = "predictions"
        split_file_name = "split.json"
        description = "test-only predictions (default)"
    else:
        pred_dir_name = "predictions_with_valid"
        split_file_name = "split_with_valid.json"
        description = "predictions with validation observations"

    log.info(f"Preparing {description} for VCB evaluation: {task_id}")
    log.info(f"  Predictions: {predictions_dir}")
    log.info(f"  Ground truth: {ground_truth_dir}")
    log.info(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and optionally filter metadata
    metadata_file = predictions_dir / "prepared_metadata.parquet"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Required metadata file not found: {metadata_file}")

    import polars as pl
    df = pl.read_parquet(metadata_file)

    if test_only:
        # Filter to test observations only
        split_values = df["split"].unique().to_list()
        test_splits = [s for s in split_values if s.startswith("test")]

        if test_splits:
            original_count = len(df)
            df = df.filter(pl.col("split").is_in(test_splits))
            log.info(f"  Filtered to test observations: {len(df)}/{original_count}")
            log.info(f"  Test splits: {test_splits}")
        else:
            log.info("  No test splits found - using all observations")

    # Create predictions directory
    pred_output = output_dir / pred_dir_name
    pred_output.mkdir(parents=True, exist_ok=True)

    # Save metadata (filtered or full)
    obs_file = pred_output / "obs.parquet"
    df.write_parquet(obs_file)

    # Source files
    features_file = predictions_dir / "features" / "pred_phenom.zarr"
    if not features_file.exists():
        raise FileNotFoundError(f"Required features file not found: {features_file}")

    # Create symlink to features
    features_link = pred_output / "features.zarr"
    if features_link.is_symlink() or features_link.exists():
        features_link.unlink()
    features_link.symlink_to(os.path.relpath(features_file, pred_output))
    log.info(f"  Created {pred_dir_name}/ directory with obs.parquet and features.zarr")

    # Create dataset metadata
    biological_context = TASK_BIOLOGICAL_CONTEXT[task_id]
    pred_metadata = {
        "dataset_id": "predictions",
        "description": f"Predicted phenomics features ({description})",
        "version": "1.0",
        "biological_context": biological_context,
    }

    # Include lineage information if provided
    if lineage:
        pred_metadata["model_lineage"] = {
            "data_version": lineage.get("data_version"),
            "model_config": lineage.get("model_config"),
            "lineage_chain": [
                {
                    "training_dir": entry.get("training_dir"),
                    "parquet_path": entry.get("config", {})
                    .get("get_dataloaders", {})
                    .get("path"),
                    "resume_from": entry.get("config", {})
                    .get("ckpt", {})
                    .get("resume_from"),
                }
                for entry in lineage.get("lineage_chain", [])
            ],
        }

    with open(pred_output / "predictions_dataset_metadata.json", "w") as f:
        json.dump(pred_metadata, f, indent=2)
    log.info(f"  Created metadata with biological_context: {biological_context}")

    # Create var file
    var_df = pl.DataFrame({"feature_name": [f"feature_{i}" for i in range(1664)]})
    var_df.write_parquet(pred_output / "predictions_var.parquet")
    log.info("  Created predictions_var.parquet")

    # Create split.json (using filtered data if test_only)
    _create_split_json(df, output_dir, ground_truth_dir, split_file_name)

    return output_dir


def _create_split_json(
    df: pl.DataFrame,
    output_dir: Path,
    ground_truth_dir: Optional[Path],
    split_file_name: str = "split.json"
) -> Path:
    """Create split.json from metadata split column with ground truth index mapping.

    Args:
        df: Predictions metadata dataframe (potentially filtered)
        output_dir: Where to save split file
        ground_truth_dir: VCB ground truth directory for index mapping
        split_file_name: Name of split file to create
    """
    log.info(f"Creating {split_file_name} from metadata...")

    split_values = df["split"].unique().to_list()
    log.info(f"  Found split values: {split_values}")

    # Identify split categories
    train_splits = [s for s in split_values if s == "train"]
    valid_splits = [s for s in split_values if s.startswith("valid")]
    test_splits = [s for s in split_values if s.startswith("test")]

    log.info(f"  Train splits: {train_splits}")
    log.info(f"  Valid splits: {valid_splits}")
    log.info(f"  Test splits: {test_splits}")

    # Handle split mapping and ground truth index correction
    if ground_truth_dir:
        split_data = _create_corrected_split_with_gt_mapping(
            df, ground_truth_dir, valid_splits, test_splits
        )
    else:
        # Fallback to prediction-based indices (may cause VCB validation errors)
        log.warning("No ground truth directory provided - using prediction indices")
        split_data = _create_prediction_based_split(df, valid_splits, test_splits)

    split_file = output_dir / split_file_name
    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=2)

    log.info(f"  Split summary:")
    log.info(f"    finetune: {len(split_data['folds'][0]['finetune'])}")
    log.info(f"    test: {len(split_data['folds'][0]['test'])}")
    log.info(f"    controls: {len(split_data['controls'])}")
    log.info(f"    base_states: {len(split_data['base_states'])}")

    return split_file


def _create_corrected_split_with_gt_mapping(
    df: pl.DataFrame,
    ground_truth_dir: Path,
    valid_splits: list[str],
    test_splits: list[str]
) -> dict:
    """Create split with proper ground truth row index mapping."""
    log.info("  Creating corrected split with ground truth index mapping...")

    # Load ground truth to get proper row indices
    gt_files = list(ground_truth_dir.glob("*_obs.parquet"))
    if not gt_files:
        raise FileNotFoundError(f"No *_obs.parquet file found in {ground_truth_dir}")

    gt_file = gt_files[0]
    log.info(f"  Loading ground truth: {gt_file}")
    gt_obs = pl.read_parquet(gt_file).with_row_index('gt_row_index')

    # Join predictions with ground truth to get row index mapping
    merged = df.join(gt_obs, on='obs_id', how='inner', suffix='_gt')

    if len(merged) != len(df):
        missing = len(df) - len(merged)
        log.warning(f"  {missing} prediction obs_ids not found in ground truth")

    # Map validation perturbations to finetune, test perturbations to test
    # Exclude controls and base states from regular splits to avoid overlap
    finetune_indices = merged.filter(
        pl.col("split").is_in(valid_splits) &
        ~pl.col("is_negative_control_gt") &
        ~pl.col("is_base_state_gt")
    )["gt_row_index"].to_list()

    test_indices = merged.filter(
        pl.col("split").is_in(test_splits) &
        ~pl.col("is_negative_control_gt") &
        ~pl.col("is_base_state_gt")
    )["gt_row_index"].to_list()

    # Controls and base states are separate categories
    control_indices = merged.filter(pl.col("is_negative_control_gt"))["gt_row_index"].to_list()

    base_state_indices = []
    if "is_base_state_gt" in merged.columns:
        base_state_indices = merged.filter(pl.col("is_base_state_gt"))["gt_row_index"].to_list()

    return {
        "dataset_id": "predictions",
        "version": 1,
        "splitting_level": "compound",
        "splitting_strategy": "random",
        "folds": [{
            "outer_fold": 0,
            "inner_fold": 0,
            "finetune": finetune_indices,
            "test": test_indices,
        }],
        "controls": control_indices,
        "base_states": base_state_indices,
    }


def _create_prediction_based_split(
    df: pl.DataFrame,
    valid_splits: list[str],
    test_splits: list[str]
) -> dict:
    """Create split using prediction zarr_index (fallback method)."""
    log.warning("  Using prediction-based indices - may cause VCB validation errors")

    # Map validation perturbations to finetune, test perturbations to test
    # Exclude controls and base states from regular splits
    finetune_indices = df.filter(
        pl.col("split").is_in(valid_splits) &
        ~pl.col("is_negative_control") &
        (~pl.col("is_base_state") if "is_base_state" in df.columns else True)
    )["zarr_index"].to_list()

    test_indices = df.filter(
        pl.col("split").is_in(test_splits) &
        ~pl.col("is_negative_control") &
        (~pl.col("is_base_state") if "is_base_state" in df.columns else True)
    )["zarr_index"].to_list()

    control_indices = df.filter(pl.col("is_negative_control"))["zarr_index"].to_list()

    base_state_indices = []
    if "is_base_state" in df.columns:
        base_state_indices = df.filter(pl.col("is_base_state"))["zarr_index"].to_list()

    return {
        "dataset_id": "predictions",
        "version": 1,
        "splitting_level": "compound",
        "splitting_strategy": "random",
        "folds": [{
            "outer_fold": 0,
            "inner_fold": 0,
            "finetune": finetune_indices,
            "test": test_indices,
        }],
        "controls": control_indices,
        "base_states": base_state_indices,
    }




def print_vcb_command(
    eval_dir: Path,
    ground_truth_dir: Path,
    task_id: str,
    pred_dir_name: str = "predictions",
    split_file_name: str = "split.json",
    note: str = "",
) -> None:
    """Print the VCB command to run evaluation."""
    pred_path = eval_dir / pred_dir_name
    split_path = eval_dir / split_file_name
    results_name = pred_dir_name.replace("predictions", "results")
    results_path = eval_dir / results_name

    print(f"\nVCB evaluation command{note}:")
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
