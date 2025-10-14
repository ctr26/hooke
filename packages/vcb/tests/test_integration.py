from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from vcb._cli.evaluate.px_cli import px_evaluate_cli
from vcb._cli.evaluate.tx_cli import tx_evaluate_cli
from vcb.data_models.dataset.metadata import DatasetMetadata
from vcb.data_models.split import Fold, Split


def mock_genetic_perturbation():
    # Intentionally incorrectly ordered
    return [
        {
            "type": "genetic",
            "ensembl_gene_id": "ENSG000000000000000000",
            "hours_post_reference": 0,
        },
    ]


def mock_compound_perturbation(pert: str):
    # Intentionally incorrectly ordered
    return [
        {
            "type": "compound",
            "inchikey": pert,
            "concentration": 1.0,
            "hours_post_reference": 1,
        }
    ] + mock_genetic_perturbation()


def mock_var():
    return pl.DataFrame({"ensembl_gene_id": [f"ENSG{i:012d}" for i in range(128)]})


@pytest.fixture(scope="function")
def mock_predictions(tmpdir):
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
    mock_var().write_parquet(var_path)

    # Create the obs
    obs = pl.DataFrame(
        {
            "perturbations": [mock_compound_perturbation("A"), mock_compound_perturbation("B")],
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
def mock_dataset(tmpdir):
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
    mock_var().write_parquet(var_path)

    # Create the obs
    obs = pl.DataFrame(
        {
            "perturbations": [
                mock_compound_perturbation("A"),
                mock_compound_perturbation("B"),
                mock_genetic_perturbation(),
                mock_genetic_perturbation(),
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
def mock_split(tmpdir):
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
    return p


def assert_perfect_performance(
    results: pl.DataFrame,
    correlation_metrics: list[str],
    retrieval_metrics: list[str],
    error_metrics: list[str],
):
    summary = (
        results.group_by("metric")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("metric")
    )

    # Correlation metrics and Retrieval metrics need to be maximized
    for metric in correlation_metrics + retrieval_metrics:
        assert np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 1.0)

    # Error metrics need to be minimized
    for metric in error_metrics:
        assert np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 0.0)


def test_evaluate_tx_cli(mock_dataset, mock_predictions, mock_split, tmpdir):
    # For Tx, predictions are assumed to be log1p-transformed.
    root = zarr.open(Path(mock_predictions) / "features.zarr", mode="r+")
    root["predictions"][:] = np.log1p(root["predictions"][:])

    results = tx_evaluate_cli(
        predictions_path=mock_predictions,
        ground_truth_path=mock_dataset,
        split_path=mock_split,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_predictions) / "var.parquet",
        predictions_zarr_index_column=None,
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine", "pearson", "pearson_delta", "cosine_delta"],
        retrieval_metrics=["retrieval_mae", "retrieval_mae_delta", "retrieval_edistance"],
        error_metrics=["mse"],
    )


def test_evaluate_px_cli(mock_dataset, mock_predictions, mock_split, tmpdir):
    results = px_evaluate_cli(
        predictions_path=mock_predictions,
        ground_truth_path=mock_dataset,
        split_path=mock_split,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_predictions) / "var.parquet",
        predictions_zarr_index_column=None,
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine"],
        retrieval_metrics=["retrieval_mae", "retrieval_edistance"],
        error_metrics=["mse"],
    )
