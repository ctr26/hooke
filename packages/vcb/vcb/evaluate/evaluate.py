import numpy as np

from vcb.metrics import DISTRIBUTIONAL_METRICS, SAMPLE_METRICS


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
