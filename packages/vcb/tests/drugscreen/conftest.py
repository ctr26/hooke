from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from tests.drugscreen.utils import mock_drugscreen_compound_perturbation, mock_drugscreen_genetic_perturbation
from tests.utils import mock_transcriptomics_var
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.dataset.metadata import DatasetMetadata
from vcb.data_models.dataset.predictions import PredictionPaths
from vcb.data_models.split import Fold, Split


@pytest.fixture(scope="function")
def mock_drugscreen_predictions_path(tmpdir):
    """
    Create minimally, mocked prediction with just a single datapoint.
    """
    dst = tmpdir.mkdir("predictions__test__v1_0")
    features_path = str(dst.join("features.zarr"))
    obs_path = str(dst.join("obs.parquet"))
    var_path = str(dst.join("var.parquet"))

    # Create the features
    root = zarr.open(features_path, mode="w")
    arr = root.create_array(name="predictions", shape=(2, 128), dtype=np.float32)
    arr[:] = np.arange(2 * 128).reshape(2, 128) * 10

    # Create the var
    mock_transcriptomics_var(128).write_parquet(var_path)

    # Create the obs
    obs = pl.DataFrame(
        {
            "perturbations": [
                mock_drugscreen_compound_perturbation("A"),
                mock_drugscreen_compound_perturbation("B"),
            ],
            "obs_id": [0, 1],
            "batch_center": ["plate_1", "plate_1"],
            "plate_disease_model": ["ENSG000000000000000000", "ENSG000000000000000000"],
            "is_base_state": [False, False],
            "drugscreen_query": [True, True],
            "cell_type": ["HUVEC", "HUVEC"],
        }
    )
    obs.write_parquet(obs_path)

    return str(dst)


@pytest.fixture
def mock_drugscreen_dataset_path(tmpdir):
    """
    Create a mock dataset with two datapoints.
    For simplicity, it contains both the ground truth and predictions.
    This is used to do an integration test where we expect perfect performance scores.
    """

    dataset_id = "dataset__test__v1_0"
    dst = tmpdir.mkdir(dataset_id)
    features_path = str(dst.join(f"{dataset_id}_features.zarr"))
    obs_path = str(dst.join(f"{dataset_id}_obs.parquet"))
    var_path = str(dst.join(f"{dataset_id}_var.parquet"))
    metadata_path = str(dst.join(f"{dataset_id}_dataset_metadata.json"))

    # Create the features
    # For a dataset, we just create a Zarr array without group structure.
    arr = zarr.create_array(store=features_path, shape=(4, 128), dtype=np.float32)
    arr[:] = np.arange(4 * 128).reshape(4, 128) * 10

    # Create the var
    mock_transcriptomics_var(128).write_parquet(var_path)

    # Create the obs
    obs = pl.DataFrame(
        {
            "perturbations": [
                mock_drugscreen_compound_perturbation("A"),
                mock_drugscreen_compound_perturbation("B"),
                mock_drugscreen_genetic_perturbation(),
                mock_drugscreen_genetic_perturbation(),
            ],
            "obs_id": [0, 1, 2, 3],
            "batch_center": ["plate_1", "plate_1", "plate_1", "plate_1"],
            "plate_disease_model": [
                "ENSG000000000000000000",
                "ENSG000000000000000000",
                "ENSG000000000000000000",
                "ENSG000000000000000000",
            ],
            "is_base_state": [False, False, True, True],
            "drugscreen_query": [True, True, False, False],
            "cell_type": ["HUVEC", "HUVEC", "HUVEC", "HUVEC"],
        }
    )
    obs.write_parquet(obs_path)

    # A dataset also needs metadata, which predictions do not need.
    metadata = DatasetMetadata(dataset_id=dataset_id, biological_context={"cell_type"})
    with open(metadata_path, "w") as f:
        f.write(metadata.model_dump_json())

    return str(dst)


@pytest.fixture
def mock_drugscreen_split_path(tmpdir):
    split = Split(
        dataset_id="dataset__test__v1_0",
        version=1,
        splitting_level="random",
        splitting_strategy="random",
        controls=[],
        base_states=[2, 3],
        folds=[Fold(outer_fold=0, inner_fold=0, finetune=[], test=[0, 1], validation=[])],
    )
    p = tmpdir.mkdir("split").join("split.json")
    with open(p, "w") as f:
        f.write(split.model_dump_json())
    return str(p)


@pytest.fixture(scope="function")
def mock_drugscreen_predictions(mock_drugscreen_predictions_path, mock_drugscreen_dataset):
    predictions = AnnotatedDataMatrix(
        **PredictionPaths(root=mock_drugscreen_predictions_path).model_dump(),
        var_path=Path(mock_drugscreen_predictions_path) / "var.parquet",
        metadata_path=mock_drugscreen_dataset.metadata_path,
        features_layer="predictions",
        zarr_index_column=None,
    )
    return predictions


@pytest.fixture(scope="function")
def mock_drugscreen_dataset(mock_drugscreen_dataset_path):
    return AnnotatedDataMatrix(**DatasetDirectory(root=mock_drugscreen_dataset_path).model_dump())


@pytest.fixture(scope="function")
def mock_drugscreen_split(mock_drugscreen_split_path):
    return Split.from_json(mock_drugscreen_split_path)
