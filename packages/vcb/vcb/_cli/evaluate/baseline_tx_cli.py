import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger

from vcb._cli.evaluate.tx_cli import tx_evaluate_cli
from vcb.baselines import BASELINES
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Split
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.preprocessing.steps.match_genes import MatchGenesStep


def run_baseline_tx_cli(
    ground_truth_path: str,
    split_file: str,
    split_idx: int,
    save_destination: Path,
    baseline_type: str,
    predictions_var_path: Path,
    library_size: int | None = None,
    train_on_validation: bool = False,
    distributional_metrics: bool = True,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
):
    """
    Evaluate baseline performance on a transcriptomics dataset.

    Args:
        ground_truth_path: Path to the root directory of the dataset.
        split_file: Path to the split json file.
        split_idx: Index of the split to evaluate.
        save_destination: Path to the results parquet file.
        baseline_type: Type of baseline to evaluate.
        train_on_validation: Whether to train on the validation split.
        distributional_metrics: Whether to include distributional metrics.
        predictions_var_path: Path to the var file for the predictions.
        predictions_gene_id_column: Column of the predictions to use for the gene id.
        ground_truth_gene_id_column: Column of the ground truth to use for the gene id.
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

    # NOTE (cwognum): Since the MatchGenesStep requires predictions and ground truth to be AnnotatedDataMatrix objects
    #    we need to create a temporary predictions object. Not super happy with this.
    tmp_predictions = AnnotatedDataMatrix(**DatasetDirectory(root=ground_truth_path).model_dump())
    tmp_predictions.var_path = predictions_var_path

    step = MatchGenesStep(
        ground_truth_gene_id_column=ground_truth_gene_id_column,
        predictions_gene_id_column=predictions_gene_id_column,
    )
    ground_truth, _ = step.fit(ground_truth, tmp_predictions).transform(ground_truth, tmp_predictions)

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

    # Predictions are expected to be log1p-transformed.
    predictions = np.log1p(predictions)

    logger.info(f"Succesfully generated predictions. Shape: {predictions.shape}")

    # Write predictions to temporary directory, and call the evaluation CLI
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

        results = tx_evaluate_cli(
            predictions_path=tmpdir,
            ground_truth_path=ground_truth_path,
            split_path=split_file,
            split_idx=split_idx,
            save_destination=save_destination,
            predictions_features_layer="predictions",
            predictions_zarr_index_column=None,
            predictions_var_path=predictions_var_path,
            library_size=library_size,
            use_validation_split=False,
            distributional_metrics=distributional_metrics,
            predictions_gene_id_column=predictions_gene_id_column,
            ground_truth_gene_id_column=ground_truth_gene_id_column,
        )

    return results
