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
from vcb.preprocessing.pipeline import TranscriptomicsPreprocessingPipeline
from vcb.preprocessing.steps.log1p import InverseLog1pStep, Log1pStep
from vcb.preprocessing.steps.match_genes import MatchGenesStep
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep


def tx_evaluate_cli(
    predictions_path: Path,
    ground_truth_path: Path,
    split_path: Path,
    split_idx: int,
    save_destination: Path,
    predictions_features_layer: str,
    predictions_zarr_index_column: str,
    predictions_var_path: Path,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
    library_size: int | None = None,
    distributional_metrics: bool = True,
    use_validation_split: bool = False,
):
    """
    Evaluate predictions in Transcriptomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory. Predictions should be in log space.
        ground_truth_path: Path to the ground truth directory.
        split_path: Path to the split json file.
        split_idx: Index of the split to evaluate.
        save_destination: Path to where results should be saved.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_zarr_index_column: Column of the predictions to use for the zarr index.
        predictions_var_path: Path to the var file for the predictions.
        predictions_gene_id_column: (optional) Column of the predictions to use for the gene id.
        ground_truth_gene_id_column: (optional) Column of the ground truth to use for the gene id.
        library_size: (optional) Library size to use for the evaluation (default ground truth median library size).
        distributional_metrics: (optional) Whether to include distributional metrics.
        use_validation_split: (optional) Whether to use the validation split instead of the test split (default False).

    NOTE (cwognum): For now, this only supports the count space. We don't yet support evaluation in embedding spaces.
    """

    # Parameterization of transformations is dissabled until it is strictly needed
    # named variables retained for readibility
    # invert log1p before scaling predictions, i.e. they are in log space
    predictions_are_in_logspace = True
    # _do_ scale the library size of predictions to match ground truth or requested value
    rescale_predictions = True
    # _do_ log transform ground truth and predictions before final metric calculation
    evaluate_in_logspace = True

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
        preprocessing_pipeline=TranscriptomicsPreprocessingPipeline(
            steps=[
                MatchGenesStep(
                    ground_truth_gene_id_column=ground_truth_gene_id_column,
                    predictions_gene_id_column=predictions_gene_id_column,
                ),
                InverseLog1pStep(
                    transform_predictions=predictions_are_in_logspace,
                    transform_ground_truth=False,
                ),
                ScaleCountsStep(library_size=library_size, transform_predictions=rescale_predictions),
                Log1pStep(
                    transform_predictions=evaluate_in_logspace,
                    transform_ground_truth=evaluate_in_logspace,
                ),
            ]
        ),
        metric_suites=[
            RetrievalSuite(
                metric_labels={"retrieval_mae", "retrieval_mae_delta", "retrieval_edistance"},
                use_distributional_metrics=distributional_metrics,
            ),
            PerturbationEffectPredictionSuite(
                metric_labels={"pearson", "pearson_delta", "cosine", "cosine_delta", "mse"},
                use_distributional_metrics=distributional_metrics,
            ),
        ],
    )

    # Evaluate
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
