from unittest.mock import Mock

import polars as pl
import pytest

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.match_genes import MatchGenesStep


def test_match_gene_space():
    """Test that match_gene_space correctly finds intersection of genes and sets indices."""

    # Create mock AnnotatedDataMatrix objects
    mock_a = Mock(spec=AnnotatedDataMatrix)
    mock_b = Mock(spec=AnnotatedDataMatrix)

    # Set up the var property to return our test data
    mock_a.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"]})
    mock_b.var = pl.DataFrame({"ensembl_gene_id": ["GENE3", "GENE5", "GENE6", "GENE2"]})

    # Track calls to set_var_indices
    mock_a.set_var_indices = Mock()
    mock_b.set_var_indices = Mock()

    # Call the function
    result_a, result_b = (
        MatchGenesStep(
            ground_truth_gene_id_column="ensembl_gene_id",
            predictions_gene_id_column="ensembl_gene_id",
        )
        .fit(mock_a, mock_b)
        .transform(mock_a, mock_b)
    )

    # Verify the function returns the same objects
    assert result_a is mock_a
    assert result_b is mock_b

    # Verify set_var_indices was called with correct indices
    # Expected intersection: ["GENE2", "GENE3"]
    # In genes_a: GENE2 is at index 1, GENE3 is at index 2
    # In genes_b: GENE2 is at index 3, GENE3 is at index 0
    mock_a.set_var_indices.assert_called_once_with([1, 2])
    mock_b.set_var_indices.assert_called_once_with([3, 0])


def test_match_genes_fit_and_transform():
    """Test that MatchGenesStep correctly fits gene intersection and transforms data."""

    mock_ground_truth = Mock(spec=AnnotatedDataMatrix)
    mock_predictions = Mock(spec=AnnotatedDataMatrix)

    # Set up gene data with some overlap
    mock_ground_truth.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"]})
    mock_predictions.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3", "GENE5", "GENE6"]})

    # Track calls to set_var_indices
    mock_ground_truth.set_var_indices = Mock()
    mock_predictions.set_var_indices = Mock()

    step = MatchGenesStep()
    assert not step.fitted

    fitted_step = step.fit(mock_ground_truth, mock_predictions)
    assert fitted_step.fitted
    assert fitted_step.gene_subset == ["GENE2", "GENE3"]

    result_gt, result_pred = fitted_step.transform(mock_ground_truth, mock_predictions)
    assert result_gt is mock_ground_truth
    assert result_pred is mock_predictions

    # Verify set_var_indices was called with correct indices
    # GENE2 at index 1, GENE3 at index 2 in ground truth
    mock_ground_truth.set_var_indices.assert_called_once_with([1, 2])
    # GENE2 at index 0, GENE3 at index 1 in predictions
    mock_predictions.set_var_indices.assert_called_once_with([0, 1])


def test_match_genes_different_gene_id_columns():
    """Test that MatchGenesStep works with different gene ID column names."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    # Use different column names
    mock_gt.var = pl.DataFrame({"gene_symbol": ["GENE1", "GENE2", "GENE3"]})
    mock_pred.var = pl.DataFrame({"ensembl_id": ["GENE2", "GENE3", "GENE4"]})

    mock_gt.set_var_indices = Mock()
    mock_pred.set_var_indices = Mock()

    step = MatchGenesStep(ground_truth_gene_id_column="gene_symbol", predictions_gene_id_column="ensembl_id")

    fitted_step = step.fit(mock_gt, mock_pred)
    assert fitted_step.gene_subset == ["GENE2", "GENE3"]

    result_gt, result_pred = fitted_step.transform(mock_gt, mock_pred)
    # GENE2 at index 1, GENE3 at index 2 in ground truth
    mock_gt.set_var_indices.assert_called_once_with([1, 2])
    # GENE2 at index 0, GENE3 at index 1 in predictions
    mock_pred.set_var_indices.assert_called_once_with([0, 1])


def test_match_genes_no_intersection():
    """Test that MatchGenesStep handles case with no gene intersection."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    # No overlapping genes
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE3", "GENE4"]})

    step = MatchGenesStep()
    fitted_step = step.fit(mock_gt, mock_pred)

    # Should result in empty gene subset
    assert fitted_step.gene_subset == []


def test_match_genes_complete_overlap():
    """Test that MatchGenesStep handles case with complete gene overlap."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    # Identical gene sets
    genes = ["GENE1", "GENE2", "GENE3"]
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": genes})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": genes})

    mock_gt.set_var_indices = Mock()
    mock_pred.set_var_indices = Mock()

    step = MatchGenesStep()
    fitted_step = step.fit(mock_gt, mock_pred)
    assert fitted_step.gene_subset == genes

    result_gt, result_pred = fitted_step.transform(mock_gt, mock_pred)
    # All genes should be selected: [0, 1, 2]
    mock_gt.set_var_indices.assert_called_once_with([0, 1, 2])
    mock_pred.set_var_indices.assert_called_once_with([0, 1, 2])


def test_match_genes_with_none_values():
    """Test that MatchGenesStep handles None values in gene IDs."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    # Include None values that should be filtered out
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", None, "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3", None]})

    mock_gt.set_var_indices = Mock()
    mock_pred.set_var_indices = Mock()

    step = MatchGenesStep()
    fitted_step = step.fit(mock_gt, mock_pred)
    # Only GENE2 should be in intersection (None values filtered out)
    assert fitted_step.gene_subset == ["GENE2"]

    result_gt, result_pred = fitted_step.transform(mock_gt, mock_pred)
    # GENE2 at index 2 in ground truth
    mock_gt.set_var_indices.assert_called_once_with([2])
    # GENE2 at index 0 in predictions
    mock_pred.set_var_indices.assert_called_once_with([0])


def test_match_genes_not_fitted_error():
    """Test that transform raises error when not fitted."""

    step = MatchGenesStep()
    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    with pytest.raises(RuntimeError, match="The gene subset is not fitted"):
        step.transform(mock_gt, mock_pred)


def test_match_genes_different_gene_counts_error():
    """Test that transform raises error when datasets have different gene counts after matching."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    # Set up overlapping genes
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3"]})

    # Mock set_var_indices to simulate different final gene counts
    def mock_set_var_indices_gt(indices):
        mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE2"]})  # 1 gene

    def mock_set_var_indices_pred(indices):
        mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3"]})  # 2 genes

    mock_gt.set_var_indices = Mock(side_effect=mock_set_var_indices_gt)
    mock_pred.set_var_indices = Mock(side_effect=mock_set_var_indices_pred)

    step = MatchGenesStep()
    fitted_step = step.fit(mock_gt, mock_pred)

    with pytest.raises(ValueError, match="different numbers of genes"):
        fitted_step.transform(mock_gt, mock_pred)


def test_match_genes_fitted_property():
    """Test the fitted property behavior."""

    step = MatchGenesStep()
    assert not step.fitted

    # Manually set gene_subset to test fitted property
    step.gene_subset = ["GENE1", "GENE2"]
    assert step.fitted

    step.gene_subset = None
    assert not step.fitted
