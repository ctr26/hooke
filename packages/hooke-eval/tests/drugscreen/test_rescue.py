import numpy as np
import polars as pl
import pytest

from vcb.data_models.task.drugscreen import add_compound_perturbation_to_obs
from vcb.metrics.drugscreen.prometheus import embed_in_prometheus_space
from vcb.metrics.drugscreen.sampling import (
    compute_glyph_sample_size,
    get_stratified_sample_counts,
    sample_stratified,
)


@pytest.mark.parametrize(
    "features, p0_vector, p1_vector, expected_result",
    [
        # Case 1: Feature vector along the principle axis
        (np.array([[1, 0]]), np.array([0, 0]), np.array([1, 0]), np.array([[1.0, 0.0]])),
        # Case 2: Feature vector along the rejection axis.
        (np.array([[0, 1]]), np.array([0, 0]), np.array([1, 0]), np.array([[0.0, 1.0]])),
        # Case 3: Simple 3-4-5 triangle.
        (np.array([[4, 5]]), np.array([1, 1]), np.array([11, 1]), np.array([[0.3, 0.4]])),
    ],
)
def test_prometheus_space_embedding(features, p0_vector, p1_vector, expected_result):
    """
    Tests transformation with a simple 2D case resulting in a 3-4-5 triangle.

    Expected recentered vectors:
    - Data D_new: (3, 4)
    - Axis P: (10, 0)

    Expected Scaled Result: [0.3, 0.4]
    """
    result = embed_in_prometheus_space(features, p0_vector, p1_vector)
    np.testing.assert_allclose(result, expected_result, atol=1e-7)
    assert result.shape == (1, 2)
    assert isinstance(result, np.ndarray)


def test_prometheus_space_embedding_shape():
    # Principle axis is simply the X-axis
    # Rejection axis is simply the Y-axis

    # Healthy data at (1, 1)
    # Large variance along principle axis, but not along rejection axis
    X0 = np.random.normal((1, 1), (0.5, 0.1), (100, 2))
    v0 = X0.mean(axis=0)

    # Disease data at (10, 1)
    # Large variance along rejection axis, but not along principle axis
    X1 = np.random.normal((10, 1), (0.1, 0.5), (100, 2))
    v1 = X1.mean(axis=0)

    X = np.vstack((X0, X1))
    projected_data = embed_in_prometheus_space(X, v0, v1)

    # Split the projected data into its two clusters
    X0_projected = projected_data[:100]
    X1_projected = projected_data[100:]

    # Healthy cluster should be around 0 on the x-axis.
    assert X0_projected[:, 0].mean() == pytest.approx(0)

    # Disease cluster should be around 1
    assert X1_projected[:, 0].mean() == pytest.approx(1)

    # Along the rejection axis, the disease cluster should have a higher max and variance.
    assert X1_projected[:, 1].max() > X0_projected[:, 1].max()
    assert X1_projected[:, 1].var() > X0_projected[:, 1].var()


@pytest.mark.parametrize(
    "replicate_counts,expected_mode",
    [
        # Case 1: Single mode, no tie-breaker needed
        ([5, 5, 5, 5], 5),
        # Case 2: Two modes, tie-breaker should pick smaller value
        ([1, 1, 2, 2, 2, 3, 3, 3, 4, 4], 2),  # Modes: [2, 3], min = 2
        # Case 3: Three modes, tie-breaker should pick smallest
        ([1, 1, 1, 2, 2, 3, 3, 4, 4, 4], 1),  # Modes: [1, 2, 3], min = 1
        # Case 4: All different counts, mode is most frequent
        ([1, 2, 2, 2, 3, 4, 5], 2),  # Mode: 2
        # Case 5: Single group
        ([10], 10),
    ],
)
def test_compute_glyph_sample_size(replicate_counts, expected_mode):
    """Test that glyph sample size returns the mode of replicate counts, with min as tie-breaker."""
    # Create test data with known replicate patterns
    perturbations = [
        [
            {"type": "genetic", "ensembl_gene_id": "gene1"},
            {"type": "compound", "inchikey": f"key{idx}", "concentration": idx},
        ]
        for idx, n_replicates in enumerate(replicate_counts)
        for i in range(n_replicates)
    ]

    obs = pl.DataFrame(
        {
            "drugscreen_query": [True] * sum(replicate_counts),
            "experiment_label": ["exp1"] * sum(replicate_counts),
            "cell_type": ["A549"] * sum(replicate_counts),
            "perturbations": perturbations,
        }
    )
    obs = add_compound_perturbation_to_obs(obs)
    result = compute_glyph_sample_size(obs)
    assert result == expected_mode
    assert isinstance(result, int)


def test_get_stratified_sample_counts():
    """Test stratified sampling returns correct sample counts for each class."""
    reference_classes = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    n_per_sample = 12

    rng = np.random.RandomState(42)
    result = get_stratified_sample_counts(reference_classes, n_per_sample, rng)

    assert result["A"] == 2
    assert result["B"] == 4
    assert result["C"] == 6
    assert all(k in ["A", "B", "C"] for k in result.keys())


@pytest.mark.parametrize(
    "reference_classes, n_per_sample, expected",
    [
        (["A"] * 20 + ["B"] * 20 + ["C"] * 20, 12, {1: 4, 2: 4, 3: 4}),
        (["A"] * 10 + ["B"] * 20 + ["C"] * 30, 12, {1: 2, 2: 4, 3: 6}),
        (["A"] * 0 + ["B"] * 40 + ["C"] * 20, 12, {1: 0, 2: 8, 3: 4}),
    ],
)
def test_sample_stratified(reference_classes, n_per_sample, expected):
    """Test stratified sampling returns data with correct shape and structure."""

    n_dim = 1

    # Start off with fully balanced data.
    data_a = np.full((20, n_dim), 1.0)  # Type A samples
    data_b = np.full((20, n_dim), 2.0)  # Type B samples
    data_c = np.full((20, n_dim), 3.0)  # Type C samples
    data = np.vstack([data_a, data_b, data_c])
    classes = ["A"] * 20 + ["B"] * 20 + ["C"] * 20

    results = sample_stratified(
        data,
        classes,
        reference_classes=reference_classes,
        sample_size=10,
        n_per_sample=n_per_sample,
    )

    for result in results:
        counts = {1: 0, 2: 0, 3: 0}

        for sample in result:
            if sample[0] == 1.0:
                counts[1] += 1
            elif sample[0] == 2.0:
                counts[2] += 1
            elif sample[0] == 3.0:
                counts[3] += 1

        for k, v in counts.items():
            assert expected[k] == v, (
                f"Class {k} should have {expected[k]} samples, but got {v}. Full result:\n{result}"
            )
