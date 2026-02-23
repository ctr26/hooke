"""Inference completion validation and recovery."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr

log = logging.getLogger(__name__)

ZARR_PRIORITY = [
    "pred_phenom.zarr",
    "pred_dino.zarr",
    "pred_images.zarr",
    "real_dino.zarr",
    "real_phenom.zarr",
]


def check_completion(output_dir: Path) -> tuple[int, int]:
    """Check how many observations are complete.

    Args:
        output_dir: Inference output directory with prepared_metadata.parquet

    Returns:
        Tuple of (complete_count, total_count)
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "prepared_metadata.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pl.read_parquet(metadata_path)
    total = len(df)
    complete = df.filter(pl.col("complete"))["complete"].sum()

    return int(complete), total


def validate_zarr_consistency(output_dir: Path) -> bool:
    """Validate zarr arrays match metadata row count.

    Args:
        output_dir: Inference output directory

    Returns:
        True if consistent, False otherwise
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "prepared_metadata.parquet"
    zarr_dir = output_dir / "features"

    if not metadata_path.exists() or not zarr_dir.exists():
        return False

    df = pl.read_parquet(metadata_path)
    n_metadata = len(df)

    # Check first available zarr
    for zarr_name in ZARR_PRIORITY:
        zarr_path = zarr_dir / zarr_name
        if zarr_path.exists():
            arr = zarr.open(str(zarr_path), mode="r")
            n_zarr = arr.shape[0]
            return n_metadata == n_zarr

    return False


def _pick_zarr_array(zarr_dir: Path, zarr_name: str | None = None):
    """Pick zarr array to use for completion checking."""
    if zarr_name is not None:
        path = zarr_dir / zarr_name
        if not path.exists():
            raise FileNotFoundError(f"Requested zarr not found: {path}")
        return zarr_name, zarr.open(str(path), mode="r")

    for candidate in ZARR_PRIORITY:
        path = zarr_dir / candidate
        if path.exists():
            return candidate, zarr.open(str(path), mode="r")

    raise FileNotFoundError(
        f"No zarr arrays found in {zarr_dir}. Looked for: {ZARR_PRIORITY}"
    )


def _compute_nonzero_mask(arr, batch_rows: int = 256) -> np.ndarray:
    """Compute mask of rows with non-zero values."""
    n = arr.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    reduce_axes = tuple(range(1, arr.ndim))

    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        block = np.asarray(arr[start:end])
        row_sum = np.nansum(np.abs(block), axis=reduce_axes)
        mask[start:end] = row_sum > 0

        if (start // batch_rows) % 100 == 0:
            log.info(f"Scanned {end:,}/{n:,} rows")

    return mask


def recover_completion_status(
    output_dir: Path,
    zarr_name: str | None = None,
    batch_rows: int = 256,
    backup: bool = True,
) -> int:
    """Recover completion status from zarr arrays.

    For crashed/preempted workers that wrote to zarr but didn't update metadata.
    Rule: a row is complete if abs(zarr[row]).sum() > 0

    Args:
        output_dir: Inference output directory
        zarr_name: Specific zarr to check (default: auto-detect)
        batch_rows: Batch size for reading zarr
        backup: Whether to create backup of metadata

    Returns:
        Number of newly recovered rows
    """
    output_dir = Path(output_dir)
    zarr_dir = output_dir / "features"
    metadata_path = output_dir / "prepared_metadata.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_dir}")

    chosen_name, arr = _pick_zarr_array(zarr_dir, zarr_name)
    log.info(f"Using zarr: {chosen_name}, shape={arr.shape}")

    df = pl.read_parquet(metadata_path)

    if "zarr_index" not in df.columns or "complete" not in df.columns:
        raise ValueError("Metadata missing required columns: zarr_index, complete")

    n_df = len(df)
    n_zarr = arr.shape[0]
    if n_df != n_zarr:
        raise ValueError(f"Row count mismatch: metadata={n_df}, zarr={n_zarr}")

    log.info("Computing completion mask from zarr...")
    mask = _compute_nonzero_mask(arr, batch_rows)

    num_from_zarr = int(mask.sum())
    num_already = int(df.filter(pl.col("complete"))["complete"].sum())

    # Merge completion status
    recover_df = pl.DataFrame({
        "zarr_index": np.arange(n_zarr, dtype=np.int64),
        "complete_from_zarr": pl.Series(mask),
    })

    df_updated = (
        df.join(recover_df, on="zarr_index", how="left")
        .with_columns(
            (pl.col("complete") | pl.col("complete_from_zarr").fill_null(False))
            .alias("complete")
        )
        .drop("complete_from_zarr")
    )

    num_after = int(df_updated.filter(pl.col("complete"))["complete"].sum())
    newly_recovered = num_after - num_already

    log.info(
        f"Completion: already={num_already:,}, from_zarr={num_from_zarr:,}, "
        f"after={num_after:,}, newly_recovered={newly_recovered:,}"
    )

    # Backup and save
    if backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = metadata_path.with_suffix(f".parquet.bak-{ts}")
        shutil.copy2(metadata_path, backup_path)

    df_updated.write_parquet(metadata_path)
    log.info(f"Updated metadata: {metadata_path}")

    return newly_recovered
