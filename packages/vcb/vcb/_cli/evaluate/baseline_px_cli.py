import tempfile
from pathlib import Path

import polars as pl
import zarr
from loguru import logger

from vcb._cli.evaluate.px_cli import px_evaluate_cli
from vcb.baselines import BASELINES
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Split
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter


def run_baseline_px_cli(
    ground_truth_path: str,
    split_file: str,
    split_idx: int,
    baseline_type: str,
    save_destination: Path,
    train_on_validation: bool = False,
    distributional_metrics: bool = True,
):
    """
    Evaluate baseline performance on a phenomics embedding dataset.

    Args:
        ground_truth_dir: Path to the root directory of the dataset.
        split_file: Path to the split json file.
        split_idx: Index of the split to evaluate.
        baseline_type: Type of baseline to evaluate.
            Options: context_mean, context_sample, perturbation_mean, perturbation_sample
        save_destination: Path to the results parquet file.
        train_on_validation: Whether to train on the validation split.
        distributional_metrics: Whether to include distributional metrics.
    """

    # Extract splits.
    split = Split.from_json(split_file)
    fold = split.folds[split_idx]

    # Define split indices.
    train_indices = fold.finetune + split.base_states
    if train_on_validation:
        train_indices += fold.validation
    test_indices = fold.test

    # Load the ground truth dataset.
    ground_truth = AnnotatedDataMatrix(**DatasetDirectory(root=ground_truth_path).model_dump())

    # Init and cache baseline.
    if baseline_type not in BASELINES.keys():
        raise ValueError(f"Baseline {baseline_type} not supported. Pick one of: {list(BASELINES.keys())}")
    baseline = BASELINES[baseline_type]()

    # Training
    logger.info(f"'Training' {baseline_type} baseline on {len(train_indices)} train indices.")
    ground_truth.set_obs_indices(train_indices)
    training_task = DrugscreenTaskAdapter(dataset=ground_truth)
    training_task.prepare()
    baseline.fit(training_task)

    # Inference
    logger.info(f"Running inference on {len(test_indices)} test indices.")
    ground_truth.set_obs_indices(test_indices)
    inference_task = DrugscreenTaskAdapter(dataset=ground_truth)
    inference_task.prepare()
    predictions = baseline.predict(inference_task)

    logger.info(f"Succesfully generated predictions. Shape: {predictions.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy the test obs from the ground truth.
        obs = pl.read_parquet(ground_truth.obs_path)
        obs = obs[test_indices]
        obs.write_parquet(tmpdir / "obs.parquet")

        # Save the predictions to a Zarr group.
        root = zarr.open(tmpdir / "features.zarr", mode="w")
        root.create_array("predictions", data=predictions)

        # Evaluate the predictions.
        logger.warning(
            "Running evaluation CLI. Be aware: We will redo some of the same transformations (e.g. gene matching) "
            "and you'll see logs related to this. This is not the most efficient, but prevents us having to "
            "either duplicate code between the evaluation and baseline CLI or permanently saving predictions to disk."
        )

        px_evaluate_cli(
            predictions_path=tmpdir,
            ground_truth_path=ground_truth_path,
            split_path=split_file,
            split_idx=split_idx,
            save_destination=save_destination,
            predictions_features_layer="predictions",
            predictions_zarr_index_column=None,
            distributional_metrics=distributional_metrics,
        )
