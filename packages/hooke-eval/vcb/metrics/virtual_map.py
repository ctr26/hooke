"""
NOTE (cwognum): Basically the same as the virtual screening metrics.
  For now separating these out in separate files since it's little code and
  it simplifies adapting to application specifics later.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from vcb.metrics.utils.enrichment_factor import enrichment_factor


def map_cosine_sim_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Error metrics between the predicted and ground truth cosine similarities in the map."""
    return {
        "mse": np.mean((y_true - y_pred) ** 2),
        "mae": np.mean(np.abs(y_true - y_pred)),
    }


def map_cosine_sim_ranking(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Measure the accuracy of the ranking for any of the cosine similarities."""
    return {"spearman": spearmanr(y_true, y_pred)[0]}


def map_cosine_sim_classification(
    y_true: np.ndarray, y_pred: np.ndarray, cosine_sim_threshold: float | None = None
) -> dict[str, float]:
    """Using post-hoc classification, classify cosine similarities into hits and non-hits. Then compute classification metrics on these classes."""

    if cosine_sim_threshold is None:
        # set to best 10% are hits, i.e. .9 quantile of y_true
        cosine_sim_threshold = float(np.quantile(y_true, 0.9))

    # Post-hoc classification
    y_true_hit = y_true > cosine_sim_threshold
    y_pred_hit = y_pred > cosine_sim_threshold

    if not np.any(y_true_hit):
        return {}

    # Metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true_hit, y_pred_hit, average="binary")

    return {
        "accuracy": accuracy_score(y_true_hit, y_pred_hit),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "auprc": average_precision_score(y_true_hit, y_pred),
        "auroc": roc_auc_score(y_true_hit, y_pred),
        "enrichment_factor_5per": enrichment_factor(y_true_hit, y_pred, fraction=0.05),
    }
