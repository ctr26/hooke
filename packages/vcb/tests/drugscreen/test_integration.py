from pathlib import Path

import numpy as np
import zarr

from tests.utils import assert_perfect_performance
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
        log1p_transform_predictions=False,
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine", "pearson", "pearson_delta", "cosine_delta"],
        retrieval_metrics=["retrieval_mae", "retrieval_mae_delta", "retrieval_edistance"],
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
    )
    assert_perfect_performance(
        results,
        correlation_metrics=["cosine"],
        retrieval_metrics=["retrieval_mae", "retrieval_edistance"],
        error_metrics=["mse"],
    )
