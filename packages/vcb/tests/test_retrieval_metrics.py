import numpy as np
import pytest

from vcb.metrics.retrieval import (
    calculate_edistance_retrieval,
    calculate_mae_retrieval,
    from_distances_to_retrieval_score,
)


def test_from_distances_to_retrieval_score_perfect_match():
    """Test case where diagonal elements are lowest (perfect retrieval score = 1.0)"""
    # Create a 3x3 distance matrix where diagonal elements are lowest
    distances = np.array(
        [
            [0.1, 0.5, 0.7],  # Row 0: diagonal element (0,0) is lowest
            [0.8, 0.1, 0.6],  # Row 1: diagonal element (1,1) is lowest
            [0.9, 0.7, 0.1],  # Row 2: diagonal element (2,2) is lowest
        ]
    )

    score = from_distances_to_retrieval_score(distances)

    # Perfect retrieval should give score of 1.0
    assert np.isclose(score, 1.0), f"Expected 1.0, got {score}"


def test_from_distances_to_retrieval_score_worst_case():
    """Test case where diagonal elements are highest (worst retrieval score = 0.0)"""
    # Create a 3x3 distance matrix where diagonal elements are highest
    distances = np.array(
        [
            [0.9, 0.2, 0.1],  # Row 0: diagonal element (0,0) is highest
            [0.1, 0.9, 0.2],  # Row 1: diagonal element (1,1) is highest
            [0.2, 0.1, 0.9],  # Row 2: diagonal element (2,2) is highest
        ]
    )

    score = from_distances_to_retrieval_score(distances)

    # Worst retrieval should give score of 0.0
    assert np.isclose(score, 0.0), f"Expected 0.0, got {score}"


def test_from_distances_to_retrieval_score_mixed_case():
    """Test case with mixed rankings to verify score calculation"""
    # Create a 3x3 distance matrix with mixed rankings
    distances = np.array(
        [
            [0.5, 0.2, 0.7],  # Row 0: diagonal (0,0) is 2nd best (rank 1)
            [0.8, 0.4, 0.1],  # Row 1: diagonal (1,1) is 2nd best (rank 1)
            [0.3, 0.6, 0.9],  # Row 2: diagonal (2,2) is worst (rank 2)
        ]
    )

    score = from_distances_to_retrieval_score(distances)

    # Expected ranks: [1, 1, 2] for diagonal elements
    # Mean rank = (1 + 1 + 2) / 3 = 4/3
    # Score = 1 - (4/3) / (3-1) = 1 - (4/3) / 2 = 1 - 2/3 = 1/3
    expected_score = 1.0 - (4.0 / 3.0) / 2.0
    assert np.isclose(score, expected_score), f"Expected {expected_score}, got {score}"


def test_from_distances_to_retrieval_score_random_case():
    """Test case with random distances to ensure function handles various inputs"""
    rng = np.random.default_rng(42)

    # Create a 4x4 random distance matrix
    distances = rng.random((4, 4))

    score = from_distances_to_retrieval_score(distances)

    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0, f"Score {score} should be between 0 and 1"

    # For random distances, we expect a score around 0.5 (but not exactly)
    # This is just a sanity check that the function doesn't crash
    assert not np.isnan(score), "Score should not be NaN"
    assert not np.isinf(score), "Score should not be infinite"


@pytest.fixture
def retrieval_test_data():
    """Test data for MAE retrieval tests."""
    # 2 groups of 2 samples each
    dt = np.dtype([("inchikey", "U27"), ("concentration", float)])
    feats = np.array([[1.0, 0.0], [1.1, 0.1], [0.0, 1.0], [0.1, 1.1]])
    labels = np.array([("A", 1.0), ("A", 1.0), ("B", 2.0), ("B", 2.0)], dtype=dt)
    base = np.array([[0.5, 0.5]] * 4)
    return feats, labels, base


def test_calculate_mae_retrieval(retrieval_test_data):
    """Test that calculate_mae_retrieval function exists and has correct signature."""
    feats, labels, base = retrieval_test_data

    # Exact same array, perfect retrieval
    score, _ = calculate_mae_retrieval(feats, feats, labels, labels, base)
    assert np.isclose(score, 1.0)

    # Invert labels
    score, _ = calculate_mae_retrieval(feats, feats, labels, labels[::-1], base)
    assert np.isclose(score, 0.0)

    # Should get 0.5 score
    y_pred = np.array([[0.5, 0.5], [0.6, 0.6], [0.5, 0.5], [0.4, 0.4]])
    score, _ = calculate_mae_retrieval(feats, y_pred, labels, labels, base)
    assert np.isclose(score, 0.5)


def test_calculate_edistance_retrieval(retrieval_test_data):
    """Test E-distance retrieval with simple integer group labels."""
    feats, labels, base = retrieval_test_data

    # Exact same array, perfect retrieval
    score, _ = calculate_edistance_retrieval(feats, feats, base, labels, labels)
    assert np.isclose(score, 1.0)

    # Invert labels, worst retrieval
    score, _ = calculate_mae_retrieval(feats, feats, labels, labels[::-1], base)
    assert np.isclose(score, 0.0)

    # Should get 0.5 score
    y_pred = np.array([[0.5, 0.5], [0.6, 0.6], [0.5, 0.5], [0.4, 0.4]])
    score, _ = calculate_edistance_retrieval(feats, y_pred, base, labels, labels)
    assert np.isclose(score, 0.5)
