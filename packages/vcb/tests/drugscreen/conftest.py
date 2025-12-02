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


def _get_mocked_features(arr: zarr.Array, compounds: list[str], n_control: int, n_disease: int):
    rng = np.random.RandomState(0)
    features = []
    for _ in compounds:
        features.append(rng.normal(size=128, loc=0.5, scale=0.1))
    for _ in range(n_control):
        features.append(rng.normal(size=128, loc=0.1, scale=0.1))
    for _ in range(n_disease):
        features.append(rng.normal(size=128, loc=0.9, scale=0.1))
    arr[:] = np.vstack(features)
    return arr


def _get_mocked_obs(compounds: list[str], n_control: int, n_disease: int):
    n_drugscreen = len(compounds)
    n = n_drugscreen + n_control + n_disease

    perturbations = []
    for compound in compounds:
        perturbations.append(mock_drugscreen_compound_perturbation(compound))
    for _ in range(n_control):
        perturbations.append([])
    for _ in range(n_disease):
        perturbations.append(mock_drugscreen_genetic_perturbation())

    obs = pl.DataFrame(
        {
            "perturbations": perturbations,
            "obs_id": list(range(n)),
            "batch_center": ["plate_1"] * n,
            "plate_disease_model": ["ENSG000000000000000000"] * n,
            "is_base_state": [False] * (n_drugscreen + n_control) + [True] * n_disease,
            "is_negative_control": [False] * n_drugscreen + [True] * n_control + [False] * n_disease,
            "drugscreen_query": [True] * n_drugscreen + [False] * (n_control + n_disease),
            "cell_type": ["HUVEC"] * n,
            "experiment_label": ["exp1"] * n,
        }
    )
    return obs


@pytest.fixture(scope="function")
def mock_drugscreen_predictions_path(tmpdir):
    """
    Create minimally, mocked predictions.
    """
    dst = tmpdir.mkdir("predictions__test__v1_0")
    features_path = str(dst.join("features.zarr"))
    obs_path = str(dst.join("obs.parquet"))
    var_path = str(dst.join("var.parquet"))

    # Create the features
    root = zarr.open(features_path, mode="w")
    arr = root.create_array(name="predictions", shape=(100, 128), dtype=np.float32)
    _get_mocked_features(arr, compounds=["A"] * 25 + ["B"] * 25, n_control=25, n_disease=25)

    # Create the var
    mock_transcriptomics_var(128).write_parquet(var_path)

    # Create the obs
    obs = _get_mocked_obs(compounds=["A"] * 25 + ["B"] * 25, n_control=25, n_disease=25)
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
    arr = zarr.create_array(store=features_path, shape=(100, 128), dtype=np.float32)
    _get_mocked_features(arr, compounds=["A"] * 25 + ["B"] * 25, n_control=25, n_disease=25)

    # Create the var
    mock_transcriptomics_var(128).write_parquet(var_path)

    # Create the obs
    obs = _get_mocked_obs(compounds=["A"] * 25 + ["B"] * 25, n_control=25, n_disease=25)
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
        base_states=list(range(50, 75)),
        controls=list(range(75, 100)),
        folds=[Fold(outer_fold=0, inner_fold=0, finetune=[], test=list(range(50)), validation=[])],
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
