from pathlib import Path

import polars as pl
from loguru import logger

from vcb.data_models.config import EvaluationConfig
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.dataset.predictions import PredictionPaths
from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter


def px_evaluate_cli(
    predictions_path: Path,
    ground_truth_path: Path,
    split_path: Path,
    split_idx: int,
    save_destination: Path,
    predictions_features_layer: str,
    predictions_zarr_index_column: str,
    predictions_var_path: Path | None = None,
    distributional_metrics: bool = True,
    use_validation_split: bool = False,
):
    """
    Evaluate predictions in Phenomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory.
        ground_truth_path: Path to the ground truth directory.
        split_path: Path to the split json file.
        split_idx: Index of the split to evaluate.
        save_destination: Path to where results should be saved.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_zarr_index_column: Column of the predictions to use for the zarr index.
        predictions_var_path: (optional) Path to the var file for the predictions.
        distributional_metrics: (optional) Whether to include distributional metrics.
        use_validation_split: (optional) Whether to use the validation split instead of the test split (default False).
    """

    # Load the ground truth.
    ground_truth = AnnotatedDataMatrix(**DatasetDirectory(root=ground_truth_path).model_dump())

    # Load the predictions.
    predictions = AnnotatedDataMatrix(
        **PredictionPaths(root=predictions_path).model_dump(),
        var_path=predictions_var_path,
        metadata_path=ground_truth.metadata_path,
        features_layer=predictions_features_layer,
        zarr_index_column=predictions_zarr_index_column,
    )

    config = EvaluationConfig(
        ground_truth=DrugscreenTaskAdapter(dataset=ground_truth),
        predictions=DrugscreenTaskAdapter(dataset=predictions),
        split_path=split_path,
        split_index=split_idx,
        use_validation_split=use_validation_split,
        metric_suites=[
            RetrievalSuite(
                metric_labels={"retrieval_mae", "retrieval_edistance"},
                use_distributional_metrics=distributional_metrics,
                context_groupby_cols={"batch_center", "cell_type"},
            ),
            PerturbationEffectPredictionSuite(
                metric_labels={"cosine", "mse"},
                use_distributional_metrics=distributional_metrics,
                context_groupby_cols={"batch_center", "cell_type"},
                perturbation_groupby_cols={"inchikey", "concentration"},
            ),
        ],
    )
    results = config.execute()

    # Save the results
    save_destination.mkdir(parents=True, exist_ok=True)
    results.write_parquet(save_destination / "results.parquet")
    with open(save_destination / "config.json", "w") as f:
        # TODO (cwognum): This is not a perfect serialization, because we don't persist which dataset subclass was used.
        f.write(config.model_dump_json(indent=4))

    # Summarize the results
    summary = (
        results.group_by("metric")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("metric")
    )
    logger.info(f"Summary of results:\n{summary}")
    return results
