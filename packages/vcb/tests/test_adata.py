import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


@pytest.fixture
def anndata():
    """Create a simple AnnotatedDataMatrix for testing indexing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data: 5 cells x 10 genes
        obs_data = pl.DataFrame({"cell_id": [f"cell_{i}" for i in range(5)]})
        var_data = pl.DataFrame({"gene_id": [f"gene_{i}" for i in range(10)]})

        obs_path = temp_path / "obs.parquet"
        var_path = temp_path / "var.parquet"
        features_path = temp_path / "features.zarr"

        obs_data.write_parquet(obs_path)
        var_data.write_parquet(var_path)

        # Create interpretable features: each cell has values [0, 1, 2, ..., 9] + cell_index
        # So cell_0 has [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # cell_1 has [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # cell_2 has [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], etc.
        features = np.array([[j + i for j in range(10)] for i in range(5)])
        zarr.save(features_path, features)

        yield AnnotatedDataMatrix(obs_path=obs_path, var_path=var_path, features_path=features_path)


def test_obs_filter(anndata):
    """Test that X and obs stay in sync when obs_indices change."""
    # Filter to first 3 observations
    anndata.filter(obs_indices=np.array([0, 1, 2]))

    obs = anndata.obs
    X = anndata.X

    assert len(obs) == 3
    assert X.shape == (3, 10)
    assert obs["cell_id"].to_list() == ["cell_0", "cell_1", "cell_2"]

    # Check that X content matches the filtered observations
    # cell_0 should have [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # cell_1 should have [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # cell_2 should have [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert np.array_equal(X[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # cell_0
    assert np.array_equal(X[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # cell_1
    assert np.array_equal(X[2], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # cell_2


def test_var_filter(anndata):
    """Test that X and var are filtered together"""
    # Filter to first 5 genes
    anndata.filter(var_indices=np.array([0, 1, 2, 3, 4]))

    obs = anndata.obs
    X = anndata.X

    assert len(obs) == 5  # obs unchanged
    assert X.shape == (5, 5)  # X filtered

    # Check that X content is correct after var filtering
    # Each cell should have its original values, but only for the first 5 genes
    assert np.array_equal(X[0], [0, 1, 2, 3, 4])  # cell_0: first 5 genes
    assert np.array_equal(X[1], [1, 2, 3, 4, 5])  # cell_1: first 5 genes
    assert np.array_equal(X[2], [2, 3, 4, 5, 6])  # cell_2: first 5 genes
    assert np.array_equal(X[3], [3, 4, 5, 6, 7])  # cell_3: first 5 genes
    assert np.array_equal(X[4], [4, 5, 6, 7, 8])  # cell_4: first 5 genes


def test_combined_filter(anndata):
    """Test that both obs and var indices work together."""
    anndata.filter(
        obs_indices=np.array([0, 2, 4]),  # 3 cells
        var_indices=np.array([0, 5]),
    )  # 2 genes

    obs = anndata.obs
    X = anndata.X

    assert len(obs) == 3
    assert X.shape == (3, 2)

    # Check that X content is correct after both filters
    # cell_0 should have [0, 5] (genes 0 and 5), cell_2 should have [2, 7], cell_4 should have [4, 9]
    assert np.array_equal(X[0], [0, 5])  # cell_0: genes 0 and 5
    assert np.array_equal(X[1], [2, 7])  # cell_2: genes 0 and 5
    assert np.array_equal(X[2], [4, 9])  # cell_4: genes 0 and 5


def test_reject_invalid_shape(anndata):
    """Test that setters reject invalid anndata shape combos"""
    with pytest.raises(ValueError):
        anndata.X = anndata.X[:-1]

    with pytest.raises(ValueError):
        anndata.X = anndata.X[:, :-2]

    with pytest.raises(ValueError):
        anndata.var = anndata.var[2:]

    with pytest.raises(ValueError):
        anndata.obs = pl.concat([anndata.obs, anndata.obs])
