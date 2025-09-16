from vcb.data.preprocessing.match_genes import match_gene_space
from vcb.data.preprocessing.scale_counts import RawCountScaler
from vcb.evaluate.evaluate import evaluate
from vcb.models.anndata import AnnotatedDataMatrix
from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.predictions import PredictionPaths


def evaluate_cli(
    predictions_path: str,
    ground_truth_path: str,
    results_path: str,
    predictions_var_path: str,
    predictions_features_layer: str,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
):
    """
    Evaluate predictions in Transcriptomics against a ground truth.

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
    scaler = RawCountScaler()
    scaler.fit(ground_truth.X)
    predictions.X = scaler.transform(predictions.X, is_log1p_transformed=True)
    ground_truth.X = scaler.transform(ground_truth.X)

    # Evaluate and save the results
    results = evaluate(predictions, ground_truth)
    results.write_parquet(results_path)
