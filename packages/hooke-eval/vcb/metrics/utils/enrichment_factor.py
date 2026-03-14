import numpy as np
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
