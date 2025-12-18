import numpy as np
import pytest

from vcb.metrics.utils.enrichment_factor import enrichment_factor


@pytest.mark.parametrize(
    "y_true,y_score,fraction,expected",
    [
        # Example from the scikit-fingerprints documentation
        ([0, 0, 1], [0.1, 0.2, 0.7], 0.05, 3.0),
        # Perfect enrichment: Both actives in the top 50%
        ([1, 0, 1, 0], [0.9, 0.7, 0.8, 0.6], 0.5, 2.0),
        # Partial enrichment: Both actives in the top 50%
        ([1, 1, 0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], 0.5, 2.0),
        # Edge case: No hits at all. Should return 0.0.
        ([0, 0, 0], [0.9, 0.2, 0.1], 0.05, 0.0),
        # Edge case: All compounds are actives. Should return 1.0
        ([1, 1, 1], [0.9, 0.8, 0.7], 0.05, 1.0),
    ],
)
def test_enrichment_factor(y_true, y_score, fraction, expected):
    """Test enrichment factor metric used in virtual screening.

    Enrichment factor measures how well a scoring function ranks active compounds
    at the top. It's the ratio of active fraction in the top fraction of samples
    to the active fraction in the entire dataset.

    An enrichment factor of:
    - > 1.0 means actives are enriched in the top fraction (good ranking)
    - = 1.0 means random/no enrichment
    - < 1.0 means actives are depleted from the top fraction (poor ranking)
    - = 0.0 means no actives in the top fraction
    """
    result = enrichment_factor(np.array(y_true), np.array(y_score), fraction=fraction)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
