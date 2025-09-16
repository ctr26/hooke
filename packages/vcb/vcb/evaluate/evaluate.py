import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from vcb.evaluate.match import yield_batch_pairs, yield_compound_pairs
from vcb.evaluate.utils import add_compound_perturbation_to_obs
from vcb.metrics import DISTRIBUTIONAL_METRICS, SAMPLE_METRICS
from vcb.metrics.retrieval import calculate_edistance_retrieval, calculate_mae_retrieval
from vcb.models.anndata import AnnotatedDataMatrix
from vcb.models.dataset import Dataset


def calculate_distributional_metrics(
    pert_pred: np.ndarray, pert_true: np.ndarray, prefix: str = "distr_"
):
    """
    Calculate distributional metrics for a given set of targets and predictions.

    Args:
        targets_dict: Dictionary of targets (pert > dose > context : (a,b))
            dict value is (embedding, expression) if trained in embedding space, (expression,) if trained in gene space
        predictions_dict: Dictionary of predictions (pert > dose > context : (a,b))
            dict value is (embedding, expression) if trained in embedding space, (expression,) if trained in gene space

    Returns:
        Dictionary of metrics
    """
    return {
        f"{prefix}{m}": fn(pert_true, pert_pred).item()
        for m, fn in DISTRIBUTIONAL_METRICS.items()
    }


def calculate_aggregated_metrics(
    pert_pred: np.ndarray,
    pert_true: np.ndarray,
    base_states: np.ndarray,
    prefix: str = "aggr_",
):
    """
    Calculate aggregated metrics for a given set of targets and predictions.
    """
    metrics_dict = dict()

    pert_pred = pert_pred.mean(axis=0)
    pert_true = pert_true.mean(axis=0)
    base_states = base_states.mean(axis=0)

    for m, fn in SAMPLE_METRICS.items():
        metrics_dict[f"{prefix}{m}"] = fn(pert_true, pert_pred, base_states).item()

    return metrics_dict


def _extend_rows(rows: list[dict], scores: dict, batch_definition: dict) -> list[dict]:
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


def evaluate(predictions: AnnotatedDataMatrix, ground_truth: Dataset) -> pl.DataFrame:
    """
    Evaluate predictions against a ground truth.

    This is currently consistent across both transcriptomics and phenomics.
    """

    rows = []

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
        rows = _extend_rows(rows, scores, batch_definition)

        scores = calculate_mae_retrieval(
            samples_pred=pred,
            samples_truth=truth,
            group_labels_pred=compounds_pred,
            group_labels_truth=compounds_truth,
        )
        rows = _extend_rows(rows, scores, batch_definition)

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
        rows = _extend_rows(rows, scores, batch_definition)

        scores = calculate_distributional_metrics(pred, truth)
        rows = _extend_rows(rows, scores, batch_definition)

    return pl.DataFrame(rows)
