from unittest.mock import Mock

import numpy as np
import polars as pl

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.pipeline import PreprocessingPipeline
from vcb.preprocessing.steps.log1p import Log1pStep
from vcb.preprocessing.steps.match_genes import MatchGenesStep
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep


def test_pipeline_transformation_order():
    """Test that the order of transformations makes sense."""

    # Create mock AnnotatedDataMatrix objects with simple, interpretable data
    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    mock_gt.X = np.array([[30, 70, 10], [60, 140, 10]])
    mock_gt.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3"]})

    # Mock set_var_indices to actually filter the data
    def mock_gt_set_var_indices(indices):
        mock_gt.X = mock_gt.X[:, indices]
        mock_gt.var = mock_gt.var[indices]

    mock_gt.set_var_indices = Mock(side_effect=mock_gt_set_var_indices)

    mock_pred.X = np.array([[20, 30], [40, 60]])
    mock_pred.var = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})

    # Mock set_var_indices to actually filter the data
    def mock_pred_set_var_indices(indices):
        mock_pred.X = mock_pred.X[:, indices]
        mock_pred.var = mock_pred.var[indices]

    mock_pred.set_var_indices = Mock(side_effect=mock_pred_set_var_indices)

    # Create pipeline with steps in logical order: match genes -> scale -> log1p
    pipeline = PreprocessingPipeline(
        steps=[
            MatchGenesStep(),
            ScaleCountsStep(transform_ground_truth=True, transform_predictions=True, target_library_size=150),
        ]
    )

    # Transform the data
    result_gt, result_pred = pipeline.transform(mock_gt, mock_pred)
    assert np.array_equal(result_gt.X, np.array([[45, 105], [45, 105]]))
    assert np.array_equal(result_pred.X, np.array([[60, 90], [60, 90]]))

    # Verify the pipeline returns the same objects (in-place transformations)
    assert result_gt is mock_gt
    assert result_pred is mock_pred


def test_pipeline_serialization_deserialization(tmpdir):
    """Test that serialization and deserialization works."""

    # Create a pipeline with different step types
    original_pipeline = PreprocessingPipeline(
        steps=[
            MatchGenesStep(ground_truth_gene_id_column="gene_symbol"),
            ScaleCountsStep(library_size=1000, transform_predictions=True),
            Log1pStep(transform_ground_truth=False, transform_predictions=True),
        ]
    )

    # Serialize to dict
    with open(tmpdir / "pipeline.json", "w") as f:
        f.write(original_pipeline.model_dump_json())

    # Deserialize from dict
    with open(tmpdir / "pipeline.json", "r") as f:
        deserialized_pipeline = PreprocessingPipeline.model_validate_json(f.read())

    # Verify the deserialized pipeline is equivalent
    assert deserialized_pipeline.kind == original_pipeline.kind
