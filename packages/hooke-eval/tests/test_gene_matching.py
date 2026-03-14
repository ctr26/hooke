from unittest.mock import Mock

import polars as pl
import pytest

from vcb.data_models.dataset.anndata import TxAnnotatedDataMatrix
from vcb.preprocessing.steps.match_genes import MatchGenesStep


def test_match_gene_space():
    """Test that match_gene_space correctly finds intersection of genes and sets indices."""

    # Create mock AnnotatedDataMatrix objects
    mock_a = Mock(spec=TxAnnotatedDataMatrix)
    mock_b = Mock(spec=TxAnnotatedDataMatrix)

    # Set up the var property to return our test data
    mock_a.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"]})
    mock_b.var = pl.DataFrame({"ensembl_gene_id": ["GENE3", "GENE5", "GENE6", "GENE2"]})
    mock_a.gene_ids = ["GENE1", "GENE2", "GENE3", "GENE4"]
    mock_b.gene_ids = ["GENE3", "GENE5", "GENE6", "GENE2"]

    # Call the function
    (MatchGenesStep().fit(mock_a, mock_b).transform(mock_a, mock_b))

    # Verify set_var_indices was called with correct indices
    # Expected intersection: ["GENE2", "GENE3"]
    # In genes_a: GENE2 is at index 1, GENE3 is at index 2
    # In genes_b: GENE2 is at index 3, GENE3 is at index 0
    mock_a.filter.assert_called_once_with(var_indices=[1, 2])
    mock_b.filter.assert_called_once_with(var_indices=[3, 0])


def test_match_genes_fit_and_transform():
    """Test that MatchGenesStep correctly fits gene intersection and transforms data."""

    mock_ground_truth = Mock(spec=TxAnnotatedDataMatrix)
    mock_predictions = Mock(spec=TxAnnotatedDataMatrix)

    # Set up gene data with some overlap
    mock_ground_truth.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"]})
    mock_predictions.var = pl.DataFrame({"ensembl_gene_id": ["GENE3", "GENE5", "GENE6", "GENE2"]})
    mock_ground_truth.gene_ids = ["GENE1", "GENE2", "GENE3", "GENE4"]
    mock_predictions.gene_ids = ["GENE2", "GENE3", "GENE5", "GENE6"]

    step = MatchGenesStep()
    assert not step.fitted

    step.fit(mock_ground_truth, mock_predictions)
    assert step.fitted
    assert step.gene_subset == ["GENE2", "GENE3"]

    step.transform(mock_ground_truth, mock_predictions)

    # Verify set_var_indices was called with correct indices
    # GENE2 at index 1, GENE3 at index 2 in ground truth
    mock_ground_truth.filter.assert_called_once_with(var_indices=[1, 2])
    # GENE2 at index 0, GENE3 at index 1 in predictions
    mock_predictions.filter.assert_called_once_with(var_indices=[0, 1])


def test_match_genes_different_gene_id_columns():
    """Test that MatchGenesStep works with different gene ID column names."""

    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    # Use different column names
    mock_gt.var = pl.DataFrame({"gene_symbol": ["GENE1", "GENE2", "GENE3"]})
    mock_pred.var = pl.DataFrame({"ensembl_id": ["GENE2", "GENE3", "GENE4"]})
    mock_gt.gene_ids = ["GENE1", "GENE2", "GENE3"]
    mock_pred.gene_ids = ["GENE2", "GENE3", "GENE4"]

    step = MatchGenesStep(ground_truth_gene_id_column="gene_symbol", predictions_gene_id_column="ensembl_id")

    step = step.fit(mock_gt, mock_pred)
    step.gene_subset == ["GENE2", "GENE3"]

    step.transform(mock_gt, mock_pred)
    # GENE2 at index 1, GENE3 at index 2 in ground truth
    mock_gt.filter.assert_called_once_with(var_indices=[1, 2])
    # GENE2 at index 0, GENE3 at index 1 in predictions
    mock_pred.filter.assert_called_once_with(var_indices=[0, 1])


def test_match_genes_no_intersection():
    """Test that MatchGenesStep handles case with no gene intersection."""

    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    # No overlapping genes
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE3", "GENE4"]})
    mock_gt.gene_ids = ["GENE1", "GENE2"]
    mock_pred.gene_ids = ["GENE3", "GENE4"]

    step = MatchGenesStep()
    fitted_step = step.fit(mock_gt, mock_pred)

    # Should result in empty gene subset
    assert fitted_step.gene_subset == []


def test_match_genes_complete_overlap():
    """Test that MatchGenesStep handles case with complete gene overlap."""

    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    # Identical gene sets
    genes = ["GENE1", "GENE2", "GENE3"]
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": genes})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": genes})
    mock_gt.gene_ids = genes
    mock_pred.gene_ids = genes

    step = MatchGenesStep()
    step.fit(mock_gt, mock_pred)
    assert step.gene_subset == genes

    step.transform(mock_gt, mock_pred)
    # All genes should be selected: [0, 1, 2]
    mock_gt.filter.assert_called_once_with(var_indices=[0, 1, 2])
    mock_pred.filter.assert_called_once_with(var_indices=[0, 1, 2])


def test_match_genes_with_none_values():
    """Test that MatchGenesStep handles None values in gene IDs."""

    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    # Include None values that should be filtered out
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", None, "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3", None]})
    mock_gt.gene_ids = ["GENE1", None, "GENE2"]
    mock_pred.gene_ids = ["GENE2", "GENE3", None]

    step = MatchGenesStep()
    step.fit(mock_gt, mock_pred)
    # Only GENE2 should be in intersection (None values filtered out)
    assert step.gene_subset == ["GENE2"]

    step.transform(mock_gt, mock_pred)
    # GENE2 at index 2 in ground truth
    mock_gt.filter.assert_called_once_with(var_indices=[2])
    # GENE2 at index 0 in predictions
    mock_pred.filter.assert_called_once_with(var_indices=[0])


def test_match_genes_not_fitted_error():
    """Test that transform raises error when not fitted."""

    step = MatchGenesStep()
    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    with pytest.raises(RuntimeError, match="The gene subset is not fitted"):
        step.transform(mock_gt, mock_pred)


def test_match_genes_different_gene_counts_error():
    """Test that transform raises error when datasets have different gene counts after matching."""

    mock_gt = Mock(spec=TxAnnotatedDataMatrix)
    mock_pred = Mock(spec=TxAnnotatedDataMatrix)

    # Set up overlapping genes
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3"]})
    mock_gt.gene_ids = ["GENE1", "GENE2"]
    mock_pred.gene_ids = ["GENE2", "GENE3"]

    # Mock set_var_indices to simulate different final gene counts
    def mock_set_var_indices_gt(var_indices):
        mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE2"]})  # 1 gene

    def mock_set_var_indices_pred(var_indices):
        mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE2", "GENE3"]})  # 2 genes

    mock_gt.filter = Mock(side_effect=mock_set_var_indices_gt)
    mock_pred.filter = Mock(side_effect=mock_set_var_indices_pred)

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
