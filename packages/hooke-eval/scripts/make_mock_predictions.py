"""makes a mock predictions directory from a dataset by applying split; code testing purposes only"""

import typer
from pathlib import Path
from vcb.data_models.split import Split
import numpy as np
import polars as pl
import zarr
import shutil


def make_mock_predictions_cli(
    dataset_path: Path,
    split_path: Path,
    output_path: Path,
    split_idx: int = 0,
    use_validation_split: bool = False,
    log1p_transform_predictions: bool = False,
):
    """makes a mock predictions directory from a dataset by applying split; code testing purposes only

    Args:
        dataset_path: Path to the dataset directory.
        split_path: Path to the split json file.
        output_path: Path to the output directory.
        split_idx: Index of the split to evaluate.
        use_validation_split: Whether to use the validation split instead of the test split.
        log1p_transform_predictions: Whether to log1p transform the predictions. Use this for transcriptomics not phenomics.
    """

    dsid = dataset_path.stem
    input_obs = dataset_path / f"{dsid}_obs.parquet"
    input_features = dataset_path / f"{dsid}_features.zarr"
    # Load the split to filter down the ground truth.
    split = Split.from_json(split_path)

    fold = split.folds[split_idx]

    if use_validation_split:
        split_indices = fold.validation + split.base_states + split.controls
    else:
        split_indices = fold.test + split.base_states + split.controls

    split_indices = np.sort(split_indices)

    # setup output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # apply split to obs and features
    obs = pl.read_parquet(input_obs)
    obs = obs[split_indices, :]
    obs = obs.with_row_index("zarr_index_generated_raw_counts")
    obs.write_parquet(f"{output_path}/obs.parquet")
    features = zarr.open(input_features)
    features = features[:]

    if log1p_transform_predictions:
        features = np.log1p(features[split_indices, :])
    else:
        features = features[split_indices, :]

    z = zarr.create_array(
        store=f"{output_path}/features.zarr",
        shape=features.shape,
        chunks=tuple([1] + list(features.shape[1:])),
        dtype=features.dtype,
    )
    z[:] = features

    # copy var as is
    shutil.copy(dataset_path / f"{dsid}_var.parquet", output_path / "var.parquet")


if __name__ == "__main__":
    typer.run(make_mock_predictions_cli)
