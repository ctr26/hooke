import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target


def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, fraction=0.05) -> float:
    """
    To prevent heavy dependencies, this implementation is copied over from:

    scikit-fingerprints's `enrichment_factor`:
    https://github.com/scikit-fingerprints/scikit-fingerprints/blob/5eb50a00b89377a0b40eed7e03c6b78da8a8550b/skfp/metrics/virtual_screening.py#L18-L95

    And RDKit's `CalcEnrichment`:
    https://github.com/rdkit/rdkit/blob/bc4fffda7b501709ebe5d4f1b5d7f6663b65fea9/rdkit/ML/Scoring/Scoring.py#L141-L170

    With slight adaptations to simplify the code for our specific use case.
    """

    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError(f"Enrichment factor is only defined for binary y_true, got {y_type}")

    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    if fraction > 1 or fraction < 0:
        raise ValueError(f"Fraction must be between 0 and 1, found {fraction}")

    num_actives = np.sum(y_true)
    if num_actives == 0:
        return 0.0

    # Look at the top fraction of the scores
    scores = sorted(zip(y_score, y_true, strict=False), reverse=True)
    num_samples = int(np.ceil(len(scores) * fraction))
    sample = scores[:num_samples]

    # Compute the number of hits in the subset
    n_active_sample = np.sum([hit for _, hit in sample])
    active_fraction_sample = n_active_sample / num_samples
    active_fraction_total = num_actives / len(scores)
    enrichment = active_fraction_sample / active_fraction_total

    return enrichment


def hit_score_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Error metrics between the predicted and ground truth hit scores."""
    return {
        "mse": np.mean((y_true - y_pred) ** 2),
        "mae": np.mean(np.abs(y_true - y_pred)),
    }


def hit_ranking(
    y_true: np.ndarray, y_pred: np.ndarray, hit_threshold: float | None = None
) -> dict[str, float]:
    """Measure the accuracy of the ranking for any of the hits."""

    if hit_threshold is None:
        # set to best 10% are hits, i.e. .9 quantile of y_true
        hit_threshold = float(np.quantile(y_true, 0.9))

    return {
        "spearman": spearmanr(y_true, y_pred)[0],
        "auprc": average_precision_score(y_true > hit_threshold, y_pred),
        "auroc": roc_auc_score(y_true > hit_threshold, y_pred),
        "enrichment_factor_5per": enrichment_factor(y_true > hit_threshold, y_pred, fraction=0.05),
    }


def hit_classification(
    y_true: np.ndarray, y_pred: np.ndarray, hit_threshold: float | None = None
) -> dict[str, float]:
    """Using post-hoc classification, classify hit scores into hits and non-hits. Then compute classification metrics on these classes."""

    if hit_threshold is None:
        # set to best 10% are hits, i.e. .9 quantile of y_true
        hit_threshold = float(np.quantile(y_true, 0.9))
    # Post-hoc classification
    y_true_hit = y_true >= hit_threshold
    y_pred_hit = y_pred >= hit_threshold

    if not np.any(y_true_hit):
        return {}

    # Metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true_hit, y_pred_hit, average="binary")
    accuracy = accuracy_score(y_true_hit, y_pred_hit)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
