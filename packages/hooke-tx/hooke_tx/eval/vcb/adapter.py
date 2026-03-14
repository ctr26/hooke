"""Adapter to convert hooke-tx inference output to VCB format for evaluation.

Uses make_mock_predictions logic: given dataset_path and split_path, create
the full output structure (obs, base_states+controls from GT), then overwrite
only the fold.test portion of features.zarr with our predictions.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import zarr

if TYPE_CHECKING:
    from hooke_tx.data.dataset import TaskDataset


def _import_vcb():
    """Lazy import of vcb. Raises clear error if vcb extra is not installed."""
    try:
        from vcb.data_models.split import Split
        from vcb._cli.evaluate.tx_cli import tx_evaluate_cli
        
        return Split, tx_evaluate_cli
    except ImportError as e:
        raise ImportError(
            "VCB evaluation requires the vcb extra. Install with: uv sync --extra vcb"
        ) from e


class VcbEvalCache:
    """Cached static data for VCB evaluation. Built once, reused until training ends."""

    __slots__ = (
        "selected_ensembl_gene_ids",
        "split_indices",
        "pred_indices",
        "base_ctrl_set",
        "obs",
        "var_filtered",
        "features_base",
        "gene_indices",
    )

    def __init__(
        self,
        selected_ensembl_gene_ids: list[str],
        split_indices: np.ndarray,
        pred_indices: np.ndarray,
        base_ctrl_set: set[int],
        obs: pl.DataFrame,
        var_filtered: pl.DataFrame,
        features_base: np.ndarray,
        gene_indices: np.ndarray,
    ) -> None:
        self.selected_ensembl_gene_ids = selected_ensembl_gene_ids
        self.split_indices = split_indices
        self.pred_indices = pred_indices
        self.base_ctrl_set = base_ctrl_set
        self.obs = obs
        self.var_filtered = var_filtered
        self.features_base = features_base
        self.gene_indices = gene_indices


def build_vcb_eval_cache(
    dataset: TaskDataset,
    dataset_path: Path,
    split_path: Path,
    selected_ensembl_gene_ids_path: Path | str,
    *,
    split_idx: int = 0,
    use_validation_split: bool = True,
) -> VcbEvalCache:
    """Build cache of static data for VCB evaluation. Call once at start of training."""
    Split, _ = _import_vcb()

    with open(selected_ensembl_gene_ids_path) as f:
        selected_ensembl_gene_ids = [line.strip() for line in f if line.strip()]

    obs_path, features_path, var_path = _get_dataset_paths(dataset_path)
    split = Split.from_json(split_path)
    fold = split.folds[split_idx]

    if use_validation_split:
        split_indices = np.array(fold.validation + split.base_states + split.controls)
        pred_indices = np.array(fold.validation)
    else:
        split_indices = np.array(fold.test + split.base_states + split.controls)
        pred_indices = np.array(fold.test)
    split_indices = np.sort(split_indices)

    (src,) = {s for s, _ in dataset.source_index_map}
    gene_indices = dataset.gene_index_map[src]

    obs = pl.read_parquet(obs_path)
    obs = obs[split_indices, :]
    obs = obs.with_row_index("zarr_index_generated_raw_counts")

    features = zarr.open(features_path, mode="r")
    features_arr = np.asarray(features[:], dtype=np.float32)
    features_full = features_arr[split_indices, :]
    features_base = features_full[:, gene_indices].copy()

    var_filtered = _build_var_for_selected_genes(
        var_path, selected_ensembl_gene_ids, var_gene_id_column="ensembl_gene_id"
    )
    base_ctrl_set = set(split.base_states + split.controls)

    return VcbEvalCache(
        selected_ensembl_gene_ids=selected_ensembl_gene_ids,
        split_indices=split_indices,
        pred_indices=pred_indices,
        base_ctrl_set=base_ctrl_set,
        obs=obs,
        var_filtered=var_filtered,
        features_base=features_base,
        gene_indices=gene_indices,
    )


def _get_dataset_paths(dataset_path: Path) -> tuple[Path, Path, Path]:
    """Return (obs_path, features_path, var_path). Supports VCB and hooke-tx formats."""
    dataset_path = Path(dataset_path)
    dsid = dataset_path.name

    vcb_obs = dataset_path / f"{dsid}_obs.parquet"
    hooke_obs = dataset_path / "obs.parquet"

    if vcb_obs.exists():
        return (
            vcb_obs,
            dataset_path / f"{dsid}_features.zarr",
            dataset_path / f"{dsid}_var.parquet",
        )
    if hooke_obs.exists():
        return (
            hooke_obs,
            dataset_path / "X.zarr",
            dataset_path / "var.parquet",
        )
    raise FileNotFoundError(
        f"Neither VCB ({vcb_obs}) nor hooke-tx ({hooke_obs}) format found in {dataset_path}"
    )


def _build_pred_by_obs_idx(
    pred_df: pd.DataFrame,
    dataset: TaskDataset,
) -> dict[int, np.ndarray]:
    """Map obs index -> predicted_expression. Assumes single-source dataset."""
    source_index_map = dataset.source_index_map
    if len(pred_df) != len(source_index_map):
        raise ValueError(
            f"pred_df length ({len(pred_df)}) must match dataset length ({len(source_index_map)}). "
            "Ensure pred_df comes from inference on the same dataset."
        )

    sources = {src for src, _ in source_index_map}
    if len(sources) > 1:
        raise ValueError(
            "VCB adapter expects single-source dataset. "
            f"Found sources: {sources}"
        )

    result: dict[int, np.ndarray] = {}
    for i in range(len(pred_df)):
        _, src_idx = source_index_map[i]
        result[src_idx] = np.asarray(pred_df.iloc[i]["predicted_expression"], dtype=np.float32)
    return result


def _build_var_for_selected_genes(
    full_var_path: Path,
    selected_ensembl_gene_ids: list[str],
    var_gene_id_column: str = "ensembl_gene_id",
) -> pl.DataFrame:
    """Filter var to selected genes (order preserved). VCB expects ensembl_gene_id column."""
    var = pl.read_parquet(full_var_path)
    gene_list = var[var_gene_id_column].to_list()
    indices = [gene_list.index(g) for g in selected_ensembl_gene_ids]
    return var[indices, :]


def write_predictions_to_vcb_format(
    pred_df: pd.DataFrame,
    dataset: TaskDataset,
    dataset_path: Path,
    split_path: Path,
    out_dir: Path,
    *,
    split_idx: int = 0,
    use_validation_split: bool = True,
    selected_ensembl_gene_ids_path: Path | str | None = None,
    cache: VcbEvalCache | None = None,
) -> None:
    """Write pred_df to VCB predictions format using make_mock_predictions logic.

    Creates the full output structure (obs, base_states+controls from GT),
    then overwrites only the fold.test portion of features.zarr with pred_df.
    Var is filtered to selected genes to match prediction dimensions.

    Args:
        pred_df: DataFrame from run_inference with predicted_expression column.
        dataset: TaskDataset used for the predict dataloader (for source_index_map).
        dataset_path: Path to dataset directory (VCB or hooke-tx format).
        split_path: Path to VCB Split JSON.
        out_dir: Output directory for obs.parquet, features.zarr, var.parquet.
        split_idx: Index of the fold to use.
        use_validation_split: If True, use fold.validation instead of fold.test.
        selected_ensembl_gene_ids_path: Path to file with one ensembl_gene_id per line.
            Required when cache is None.
        cache: Optional prebuilt cache. When provided, skips loading static data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache is not None:
        obs = cache.obs
        var_filtered = cache.var_filtered
        split_indices = cache.split_indices
        pred_indices = cache.pred_indices
        base_ctrl_set = cache.base_ctrl_set
        features_base = cache.features_base
    else:
        if selected_ensembl_gene_ids_path is None:
            raise ValueError(
                "selected_ensembl_gene_ids_path is required when cache is not provided."
            )
        Split, _ = _import_vcb()

        with open(selected_ensembl_gene_ids_path) as f:
            selected_ensembl_gene_ids = [line.strip() for line in f if line.strip()]

        obs_path, features_path, var_path = _get_dataset_paths(dataset_path)
        split = Split.from_json(split_path)
        fold = split.folds[split_idx]

        if use_validation_split:
            split_indices = np.array(fold.validation + split.base_states + split.controls)
            pred_indices = np.array(fold.validation)
        else:
            split_indices = np.array(fold.test + split.base_states + split.controls)
            pred_indices = np.array(fold.test)
        split_indices = np.sort(split_indices)

        (src,) = {s for s, _ in dataset.source_index_map}
        gene_indices = dataset.gene_index_map[src]

        obs = pl.read_parquet(obs_path)
        obs = obs[split_indices, :]
        obs = obs.with_row_index("zarr_index_generated_raw_counts")

        features = zarr.open(features_path, mode="r")
        features_arr = np.asarray(features[:], dtype=np.float32)
        features_full = features_arr[split_indices, :]
        features_base = features_full[:, gene_indices].copy()

        var_filtered = _build_var_for_selected_genes(
            var_path, selected_ensembl_gene_ids, var_gene_id_column="ensembl_gene_id"
        )
        base_ctrl_set = set(split.base_states + split.controls)

    pred_by_obs_idx = _build_pred_by_obs_idx(pred_df, dataset)

    features_split = features_base.copy()
    for k in range(len(pred_indices)):
        obs_idx = pred_indices[k]
        if obs_idx in base_ctrl_set:
            continue
        if obs_idx not in pred_by_obs_idx:
            continue
        out_pos = np.searchsorted(split_indices, obs_idx)
        if out_pos >= len(features_split):
            raise IndexError(f"obs_idx {obs_idx} not in split_indices")
        features_split[out_pos, :] = np.maximum(pred_by_obs_idx[obs_idx], 0)

    obs.write_parquet(out_dir / "obs.parquet")
    var_filtered.write_parquet(out_dir / "var.parquet")

    features_zarr_path = out_dir / "features.zarr"
    if features_zarr_path.exists():
        shutil.rmtree(features_zarr_path)
    
    z = zarr.create_array(
        store=str(features_zarr_path),
        shape=features_split.shape,
        chunks=(1,) + features_split.shape[1:],
        dtype=features_split.dtype,
    )
    z[:] = features_split


def evaluate_with_vcb(
    predictions_path: Path,
    ground_truth_path: Path,
    split_path: Path,
    save_destination: Path,
    predictions_var_path: Path,
    task_id: str = "phenorescue",
    *,
    split_idx: int = 0,
    use_validation_split: bool = False,
    predictions_zarr_index_column: str = "zarr_index_generated_raw_counts",
    predictions_features_layer: str | None = None,
    distributional_metrics: bool = True,
) -> pl.DataFrame:
    """Run VCB evaluation on predictions (reuses VCB tx_cli logic).

    Call this after write_predictions_to_vcb_format. Requires ground truth
    and split in VCB format.
    """
    _, tx_evaluate_cli = _import_vcb()

    return tx_evaluate_cli(
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
        split_path=split_path,
        save_destination=save_destination,
        predictions_var_path=predictions_var_path,
        task_id=task_id,
        split_idx=split_idx,
        use_validation_split=use_validation_split,
        predictions_zarr_index_column=predictions_zarr_index_column,
        predictions_features_layer=predictions_features_layer,
        distributional_metrics=distributional_metrics,
    )


def run_vcb_eval_with_temp_dir(
    pred_df: pd.DataFrame,
    dataset: TaskDataset,
    dataset_path: Path,
    split_path: Path,
    selected_ensembl_gene_ids_path: Path | str,
    ground_truth_path: Path | None = None,
    *,
    split_idx: int = 0,
    use_validation_split: bool = True,
    task_id: str = "phenorescue",
    distributional_metrics: bool = True,
    cache: VcbEvalCache | None = None,
    persistent_dir: Path | None = None,
) -> dict[str, float]:
    """Run full VCB eval: write predictions, evaluate, optionally cleanup.

    When cache and persistent_dir are provided, reuses cached data and a persistent
    directory (no create/delete per eval). Otherwise uses a temp dir.

    Returns metrics dict {metric_name: mean_score} for logging.

    Args:
        pred_df: DataFrame from run_inference.
        dataset: TaskDataset used for predict dataloader.
        dataset_path: Path to dataset directory.
        split_path: Path to VCB Split JSON.
        selected_ensembl_gene_ids_path: Path to file with one ensembl_gene_id per line.
        ground_truth_path: Path to ground truth (default: same as dataset_path).
        cache: Optional prebuilt cache. When set, skips loading static data.
        persistent_dir: Optional persistent directory. When set, reused instead of temp.
    """
    ground_truth_path = ground_truth_path or dataset_path

    if persistent_dir is not None:
        tmp = Path(persistent_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        pred_dir = tmp / "predictions"
        save_dir = tmp / "results"
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        tmpdir = tempfile.mkdtemp(prefix="hooke_vcb_eval_")
        tmp = Path(tmpdir)
        pred_dir = tmp / "predictions"
        save_dir = tmp / "results"
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        use_temp = True

    try:
        write_predictions_to_vcb_format(
            pred_df,
            dataset,
            dataset_path,
            split_path,
            pred_dir,
            split_idx=split_idx,
            use_validation_split=use_validation_split,
            selected_ensembl_gene_ids_path=selected_ensembl_gene_ids_path,
            cache=cache,
        )

        results = evaluate_with_vcb(
            predictions_path=pred_dir,
            ground_truth_path=ground_truth_path,
            split_path=split_path,
            save_destination=save_dir,
            predictions_var_path=pred_dir / "var.parquet",
            task_id=task_id,
            split_idx=split_idx,
            use_validation_split=use_validation_split,
            distributional_metrics=distributional_metrics,
        )

        summary = results.group_by("metric").agg(pl.col("score").mean().alias("mean"))
        return {d["metric"]: float(d["mean"]) for d in summary.to_dicts()}
    finally:
        if use_temp:
            shutil.rmtree(tmp, ignore_errors=True)


def create_vcb_persistent_dir() -> Path:
    """Create a persistent directory for VCB eval. Call cleanup_vcb_persistent_dir when done."""
    path = Path(tempfile.gettempdir()) / f"hooke_vcb_eval_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_vcb_persistent_dir(path: Path) -> None:
    """Remove persistent VCB eval directory."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
