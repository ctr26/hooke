from unittest.mock import Mock

import polars as pl

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.match_genes import match_gene_space


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
    result_a, result_b = match_gene_space(mock_a, mock_b)

    # Verify the function returns the same objects
    assert result_a is mock_a
    assert result_b is mock_b

    # Verify set_var_indices was called with correct indices
    # Expected intersection: ["GENE2", "GENE3"]
    # In genes_a: GENE2 is at index 1, GENE3 is at index 2
    # In genes_b: GENE2 is at index 3, GENE3 is at index 0
    mock_a.set_var_indices.assert_called_once_with([1, 2])
    mock_b.set_var_indices.assert_called_once_with([3, 0])
