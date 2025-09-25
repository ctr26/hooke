import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from vcb.evaluate.match import yield_compound_pairs
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


def _extend_rows(rows: list[dict], scores: dict, context: dict) -> list[dict]:
    """Helper function to extend the rows list with the scores and batch definition."""
    for k, v in scores.items():
        if isinstance(v, list):
            for i, v_i in enumerate(v):
                rows.append(
                    {
                        **context,
                        "score": v_i,
                        "metric": k,
                        "sample_index": i,
                    }
                )
        else:
            rows.append({**context, "score": v, "metric": k})
    return rows


def evaluate(
    predictions: AnnotatedDataMatrix,
    ground_truth: Dataset,
    distributional_metrics: bool = True,
) -> pl.DataFrame:
    """
    Evaluate predictions against a ground truth.

    This is currently consistent across both transcriptomics and phenomics.
    """

    rows = []

    logger.info("Calculating retrieval metrics.")

    # Prepare the observation-level metadata
    truth_obs = ground_truth.obs.with_row_index("original_index")
    predictions_obs = add_compound_perturbation_to_obs(
        predictions.obs.with_row_index("original_index")
    )
    total = predictions_obs["plate_disease_model"].n_unique()

    # Loop over each disease model.
    for (disease_model,), predictions_group_obs in tqdm(
        predictions_obs.group_by("plate_disease_model"),
        total=total,
        desc="Calculating retrieval per disease model",
    ):
        # Filter down the ground truth to the same group.
        truth_group_obs = add_compound_perturbation_to_obs(
            truth_obs.filter(pl.col("obs_id").is_in(predictions_group_obs["obs_id"]))
        )

        # Get the actual features
        predictions_group = predictions.X[
            predictions_group_obs["original_index"].to_list()
        ]
        truth_group = ground_truth.X[truth_group_obs["original_index"].to_list()]

        # Get the group labels
        pert_pred = list(predictions_group_obs["inchikey", "concentration"].iter_rows())
        pert_truth = list(truth_group_obs["inchikey", "concentration"].iter_rows())

        if distributional_metrics:
            scores = calculate_edistance_retrieval(
                samples_pred=predictions_group,
                samples_truth=truth_group,
                group_labels_pred=pert_pred,
                group_labels_truth=pert_truth,
            )
            rows = _extend_rows(rows, scores, {"disease_model": disease_model})

        scores = calculate_mae_retrieval(
            samples_pred=predictions_group,
            samples_truth=truth_group,
            group_labels_pred=pert_pred,
            group_labels_truth=pert_truth,
        )
        rows = _extend_rows(rows, scores, {"disease_model": disease_model})

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

        if distributional_metrics:
            scores = calculate_distributional_metrics(pred, truth)
            rows = _extend_rows(rows, scores, batch_definition)

    return pl.DataFrame(rows)
