import polars as pl

from vcb.data.preprocessing.match_genes import match_gene_space
from vcb.data.preprocessing.scale_counts import RawCountScaler
from vcb.evaluate.evaluate import evaluate
from vcb.models.anndata import AnnotatedDataMatrix
from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.predictions import PredictionPaths


def tx_evaluate_cli(
    predictions_path: str,
    ground_truth_path: str,
    results_path: str,
    predictions_features_layer: str,
    predictions_var_path: str,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
    library_size: int | None = None,
    distributional_metrics: bool = True,
):
    """
    Evaluate predictions in Transcriptomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory.
        ground_truth_path: Path to the ground truth directory.
        results_path: Path to the results parquet file.
        predictions_var_path: Path to the var file for the predictions.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_gene_id_column: (optional) Column of the predictions to use for the gene id.
        ground_truth_gene_id_column: (optional) Column of the ground truth to use for the gene id.
        library_size: (optional) Library size to use for the evaluation.
        distributional_metrics: (optional) Whether to include distributional metrics.

    NOTE (cwognum): For now, this only supports the count space. We don't yet support evaluation in embedding spaces.
    """

    # Load the predictions.
    predictions = AnnotatedDataMatrix(
        **PredictionPaths(
            root=predictions_path,
            var_path=predictions_var_path,
        ).model_dump(),
        features_layer=predictions_features_layer,
    )

    # Load the ground truth.
    ground_truth = Dataset.from_directory(DatasetDirectory(root=ground_truth_path))

    # Match the gene space
    predictions, ground_truth = match_gene_space(
        predictions,
        ground_truth,
        predictions_gene_id_column,
        ground_truth_gene_id_column,
    )

    # Scale to a consistent library size
    if library_size is not None:
        scaler = RawCountScaler(desired_library_size=library_size)
    else:
        scaler = RawCountScaler()
        scaler.fit(ground_truth.X)
        predictions.X = scaler.transform(predictions.X, is_log1p_transformed=True)

    ground_truth.X = scaler.transform(ground_truth.X)

    # Evaluate and save the results
    results = evaluate(
        predictions, ground_truth, distributional_metrics=distributional_metrics
    )
    results.write_parquet(results_path)

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
    print(summary)
