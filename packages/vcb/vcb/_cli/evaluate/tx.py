import polars as pl
from loguru import logger
from tqdm import tqdm

from vcb.data.preprocessing.match_genes import match_gene_space
from vcb.data.preprocessing.scale_counts import RawCountScaler
from vcb.evaluate.evaluate import (
    calculate_aggregated_metrics,
    calculate_distributional_metrics,
)
from vcb.evaluate.match import yield_batch_pairs, yield_compound_pairs
from vcb.evaluate.utils import add_compound_perturbation_to_obs
from vcb.metrics.retrieval import calculate_edistance_retrieval, calculate_mae_retrieval
from vcb.models.anndata import AnnotatedDataMatrix
from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.predictions import PredictionPaths


def evaluate(
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

    rows = []

    def extend_rows(scores, batch_definition):
        """Helper function to extend the rows list with the scores and batch definition."""
        for k, v in scores.items():
            if isinstance(v, list):
                for i, v_i in enumerate(v):
                    rows.append(
                        {
                            **batch_definition,
                            "score": v_i,
                            "metric": k,
                            "sample_index": i,
                        }
                    )
            else:
                rows.append({**batch_definition, "score": v, "metric": k})
        return rows

    logger.info("Calculating batch-level metrics.")
    for pred, truth, base, batch_definition, compounds_pred, compounds_truth in tqdm(
        yield_batch_pairs(predictions, ground_truth),
        total=predictions.obs["batch_center"].n_unique(),
    ):
        scores = calculate_edistance_retrieval(
            samples_pred=pred,
            samples_truth=truth,
            group_labels_pred=compounds_pred,
            group_labels_truth=compounds_truth,
        )
        extend_rows(scores, batch_definition)

        scores = calculate_mae_retrieval(
            samples_pred=pred,
            samples_truth=truth,
            group_labels_pred=compounds_pred,
            group_labels_truth=compounds_truth,
        )
        extend_rows(scores, batch_definition)

    # This is not strictly needed, but makes for nicer progress bars.
    logger.info("Calculating compound-level metrics.")
    query = add_compound_perturbation_to_obs(
        predictions.obs.filter(pl.col("drugscreen_query")).filter(
            pl.col("perturbations").list.len() == 2
        )
    )
    total = query[
        "inchikey",
        "concentration",
        "batch_center",
        *ground_truth.metadata.biological_context,
    ].n_unique()

    for pred, truth, base, batch_definition in tqdm(
        yield_compound_pairs(predictions, ground_truth),
        total=total,
    ):
        scores = calculate_aggregated_metrics(pred, truth, base)
        extend_rows(scores, batch_definition)

        scores = calculate_distributional_metrics(pred, truth)
        extend_rows(scores, batch_definition)

    pl.DataFrame(rows).write_parquet(results_path)
