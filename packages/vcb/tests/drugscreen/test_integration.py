from pathlib import Path

import numpy as np
import zarr

from tests.utils import assert_imperfect_performance, assert_perfect_performance
from vcb._cli.evaluate.px_cli import px_evaluate_cli
from vcb._cli.evaluate.tx_cli import tx_evaluate_cli


def test_evaluate_tx_cli(
    mock_drugscreen_dataset_path, mock_drugscreen_predictions_path, mock_drugscreen_split_path, tmpdir
):
    """
    End to end test for Drugscreen Transcriptomics.
    """
    # For Tx, predictions are assumed to be log1p-transformed.
    root = zarr.open(Path(mock_drugscreen_predictions_path) / "features.zarr", mode="r+")

    root["predictions"][:] = np.log1p(root["predictions"][:])

    results = tx_evaluate_cli(
        predictions_path=mock_drugscreen_predictions_path,
        ground_truth_path=mock_drugscreen_dataset_path,
        split_path=mock_drugscreen_split_path,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_drugscreen_predictions_path) / "var.parquet",
        predictions_zarr_index_column=None,
        task_id="phenorescue",
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine", "pearson", "pearson_delta", "cosine_delta"],
        retrieval_metrics=["retrieval_mae", "retrieval_edistance"],
        error_metrics=["mse"],
    )

    # make sure it doesn't get perfect scores for random data
    root["predictions"][:] = np.random.uniform(0, 6, size=root["predictions"].shape)

    results = tx_evaluate_cli(
        predictions_path=mock_drugscreen_predictions_path,
        ground_truth_path=mock_drugscreen_dataset_path,
        split_path=mock_drugscreen_split_path,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results3"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_drugscreen_predictions_path) / "var.parquet",
        predictions_zarr_index_column=None,
        task_id="phenorescue",
    )
    assert_imperfect_performance(
        results,
        correlation_metrics=["cosine", "pearson", "pearson_delta", "cosine_delta"],
        retrieval_metrics=[],  # with two perts and tiny test case, retrieval could be 1 by chance
        error_metrics=["mse"],
    )


def test_rescale_tx_cli(
    mock_drugscreen_dataset_path, mock_drugscreen_predictions_path, mock_drugscreen_split_path, tmpdir
):
    """
    End to end test for Drugscreen Transcriptomics that scaling is on and scoring robust to library size
    """
    # Tests that rescaling of predictions is working and on by default.
    # This test is less because it _has_ to be on, and more to make sure any changes
    # on that point are made intentionally

    # similarly test params for predictions without rescaling
    # first: corrupt so it needs rescaling
    root = zarr.open(Path(mock_drugscreen_predictions_path) / "features.zarr", mode="r+")
    root["predictions"][:] = np.log1p(
        root["predictions"][:] * np.random.uniform(1, 2, size=(root["predictions"].shape[0], 1))
    )

    results = tx_evaluate_cli(
        predictions_path=mock_drugscreen_predictions_path,
        ground_truth_path=mock_drugscreen_dataset_path,
        split_path=mock_drugscreen_split_path,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results2"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_drugscreen_predictions_path) / "var.parquet",
        predictions_zarr_index_column=None,
        task_id="phenorescue",
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine", "pearson", "pearson_delta", "cosine_delta"],
        retrieval_metrics=["retrieval_mae", "retrieval_edistance"],
        error_metrics=["mse"],
    )


def test_evaluate_px_cli(
    mock_drugscreen_dataset_path, mock_drugscreen_predictions_path, mock_drugscreen_split_path, tmpdir
):
    """
    End to end test for Drugscreen Phenomics.
    """
    results = px_evaluate_cli(
        predictions_path=mock_drugscreen_predictions_path,
        ground_truth_path=mock_drugscreen_dataset_path,
        split_path=mock_drugscreen_split_path,
        split_idx=0,
        save_destination=Path(str(tmpdir.mkdir("results"))),
        predictions_features_layer="predictions",
        predictions_var_path=Path(mock_drugscreen_predictions_path) / "var.parquet",
        predictions_zarr_index_column=None,
        task_id="phenorescue",
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine"],
        retrieval_metrics=["retrieval_mae", "retrieval_edistance"],
        error_metrics=["mse"],
    )
