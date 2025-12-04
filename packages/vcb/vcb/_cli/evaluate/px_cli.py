from pathlib import Path

import polars as pl
import typer
from loguru import logger
from pydantic import TypeAdapter
from typing_extensions import Annotated

from vcb.data_models.config import TASK_ADAPTERS_TYPE, EvaluationConfig
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.dataset.predictions import PredictionPaths
from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.phenorescue import PhenorescueSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.metrics.suites.virtual_map import VirtualMapSuite


def get_metric_suites_for_task_id(
    task_id: str, distributional_metrics: bool, save_destination: Path
) -> list[MetricSuite]:
    suites = [
        RetrievalSuite(
            metric_labels={"retrieval_mae", "retrieval_edistance"},
            use_distributional_metrics=distributional_metrics,
        ),
        PerturbationEffectPredictionSuite(
            metric_labels={"pearson", "pearson_delta", "cosine", "cosine_delta", "mse"},
            use_distributional_metrics=distributional_metrics,
        ),
    ]

    if task_id == "phenorescue":
        rescue_suite = PhenorescueSuite(
            metric_labels={"hit_score_error", "hit_classification", "hit_ranking"},
            plot_destination=save_destination / "phenorescue",
        )
        suites.append(rescue_suite)

    elif task_id == "virtual_map":
        virtual_map_suite = VirtualMapSuite(
            metric_labels={"map_mse"},
            plot_destination=save_destination / "virtual_map",
        )
        suites.append(virtual_map_suite)

    return suites


def px_evaluate_cli(
    predictions_path: Annotated[
        Path, typer.Option(..., "--predictions-path", "-p", help="path to the predictions directory")
    ],
    ground_truth_path: Annotated[
        Path, typer.Option(..., "--ground-truth-path", "-t", help="path to the ground truth directory")
    ],
    split_path: Annotated[Path, typer.Option(..., "--split-path", "-s", help="path to the split json file")],
    save_destination: Annotated[
        Path, typer.Option(..., "--save-destination", "-o", help="path to where results should be saved")
    ],
    predictions_var_path: Annotated[
        Path,
        typer.Option(..., "--predictions-var-path", "-v", help="path to the var file for the predictions"),
    ],
    task_id: Annotated[str, typer.Option(help="The task id. Either 'phenorescue' or 'virtual_map")],
    predictions_features_layer: Annotated[
        str, typer.Option(help="layer of the features in the zarr file to use for the predictions")
    ] = None,
    predictions_zarr_index_column: Annotated[
        str,
        typer.Option(
            help="column of the predictions that corresponds to the features/predictions zarr index"
        ),
    ] = "zarr_index_generated_raw_counts",
    split_idx: Annotated[int, typer.Option(help="index of the split to evaluate")] = 0,
    distributional_metrics: Annotated[
        bool, typer.Option(help="whether to include distributional metrics (exclude to speed up evaluation)")
    ] = True,
    use_validation_split: Annotated[
        bool,
        typer.Option(
            help="whether to use the validation split instead of the test split (use to compare evaluation and fine tuning 1:1)"
        ),
    ] = False,
):
    """
    Evaluate predictions in Phenomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory.
        ground_truth_path: Path to the ground truth directory.
        split_path: Path to the split json file.
        save_destination: Path to where results should be saved.
        predictions_var_path: (optional) Path to the var file for the predictions.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_zarr_index_column: Column of the predictions to use for the zarr index.
        task_adapter: Task adapter subclass, in snake case (e.g. "drugscreen", "singles").
        split_idx: Index of the split to evaluate.
        distributional_metrics: Whether to include distributional metrics (exclude to speed up evaluation).
        use_validation_split: Whether to use the validation split instead of the test split (default False).
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

    # TaskAdapterClass = key_to_task_adapter_class(task_adapter)
    type_adapter = TypeAdapter(TASK_ADAPTERS_TYPE)

    if task_id == "phenorescue":
        task_adapter = "drugscreen"
    elif task_id == "virtual_map":
        task_adapter = "singles"
    else:
        raise ValueError(f"Unknown task id: {task_id}")

    config = EvaluationConfig(
        ground_truth=type_adapter.validate_python({"kind": task_adapter, "dataset": ground_truth}),
        predictions=type_adapter.validate_python({"kind": task_adapter, "dataset": predictions}),
        split_path=split_path,
        split_index=split_idx,
        use_validation_split=use_validation_split,
        metric_suites=get_metric_suites_for_task_id(
            task_id,
            distributional_metrics,
            save_destination,
        ),
    )
    results = config.execute()

    # Save the results
    save_destination.mkdir(parents=True, exist_ok=True)
    results.write_parquet(save_destination / "results.parquet")
    with open(save_destination / "config.json", "w") as f:
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
