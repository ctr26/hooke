import pytest

import numpy as np
import polars as pl
import tempfile
import zarr
from pathlib import Path

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.pipeline import PreprocessingPipeline
from vcb.preprocessing.steps.log1p import Log1pStep
from vcb.preprocessing.steps.match_genes import MatchGenesStep
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep


@pytest.fixture
def gt_and_pred():
    """Create a gt and pred AnnotatedDataMatrices for testing indexing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # ground truth
        obs_path = temp_path / "obs_gt.parquet"
        var_path = temp_path / "var_gt.parquet"
        features_path = temp_path / "features_gt.zarr"

        obs_data = pl.DataFrame({"obs_id": [f"obs_{i}" for i in range(2)]})
        obs_data.write_parquet(obs_path)

        var_data = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2", "GENE3"]})
        var_data.write_parquet(var_path)

        features = np.array([[30, 70, 10], [60, 140, 10]])

        zarr.save(features_path, features)

        # predictions
        obs_path_p = temp_path / "obs_p.parquet"
        var_path_p = temp_path / "var_p.parquet"
        features_path_p = temp_path / "features_p.zarr"

        obs_data_p = pl.DataFrame({"obs_id": [f"obs_{i}" for i in range(2)]})
        obs_data_p.write_parquet(obs_path_p)

        var_data_p = pl.DataFrame({"ensembl_gene_id": ["GENE1", "GENE2"]})
        var_data_p.write_parquet(var_path_p)

        features_p = np.array([[20, 30], [40, 60]])
        zarr.save(features_path_p, features_p)

        out = (
            AnnotatedDataMatrix(obs_path=obs_path, var_path=var_path, features_path=features_path),
            AnnotatedDataMatrix(obs_path=obs_path_p, var_path=var_path_p, features_path=features_path_p),
        )
        yield out


def test_pipeline_order_can_work(gt_and_pred):
    """Test that transformation pipeline works when the order of transformations makes sense."""

    # Create mock AnnotatedDataMatrix objects with simple, interpretable data
    mock_gt, mock_pred = gt_and_pred

    # Create pipeline with steps in logical order: match genes -> scale -> log1p
    pipeline = PreprocessingPipeline(
        steps=[
            MatchGenesStep(),
            ScaleCountsStep(transform_ground_truth=True, transform_predictions=True, library_size=150),
        ]
    )

    # Transform the data
    pipeline.transform(mock_gt, mock_pred)
    assert np.array_equal(mock_gt.X, np.array([[45, 105], [45, 105]]))
    assert np.array_equal(mock_pred.X, np.array([[60, 90], [60, 90]]))


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
