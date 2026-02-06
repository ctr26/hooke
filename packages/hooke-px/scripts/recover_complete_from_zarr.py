#!/usr/bin/env python3
"""Recover `complete` flags in prepared_metadata.parquet from saved Zarr outputs.

This is intended for preempted/crashed distributed inference runs where workers
wrote features to shared Zarr arrays but did not persist per-row completion to
parquet.

Rule: a row is considered complete if `abs(zarr[row]).sum() > 0`.
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

try:
    import numpy as np
    import polars as pl
    import zarr
except ModuleNotFoundError as e:
    # Allow `--help` to work even if not running inside the project env.
    np = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]
    zarr = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Add parent directory to path for imports (repo-local execution)
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


DEFAULT_ZARR_CANDIDATES = [
    "pred_phenom.zarr",
    "pred_dino.zarr",
    "pred_images.zarr",
    "real_dino.zarr",
    "real_phenom.zarr",
]


def _pick_zarr_array(zarr_dir: Path, zarr_name: Optional[str]) -> tuple[str, Any]:
    if zarr_name is not None:
        p = zarr_dir / zarr_name
        if not p.exists():
            raise FileNotFoundError(f"Requested zarr not found: {p}")
        return zarr_name, zarr.open(str(p), mode="r")

    for candidate in DEFAULT_ZARR_CANDIDATES:
        p = zarr_dir / candidate
        if p.exists():
            return candidate, zarr.open(str(p), mode="r")

    raise FileNotFoundError(
        f"No zarr arrays found in {zarr_dir}. Looked for: {DEFAULT_ZARR_CANDIDATES}"
    )


def _row_nonzero_mask(arr: Any, batch_rows: int) -> Any:
    n = int(arr.shape[0])
    mask = np.zeros(n, dtype=np.bool_)

    # Sum over all non-row axes
    reduce_axes = tuple(range(1, arr.ndim))

    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        block = np.asarray(arr[start:end])

        # Use nansum to avoid NaNs poisoning the sum.
        row_sum = np.nansum(np.abs(block), axis=reduce_axes)
        # row_sum shape: (end-start,)
        mask[start:end] = row_sum > 0

        if (start // batch_rows) % 50 == 0:
            done = end
            log.info(f"Scanned {done:,}/{n:,} rows")

    return mask


def recover_complete_flags(
    output_dir: Path,
    zarr_dir: Optional[Path] = None,
    prepared_path: Optional[Path] = None,
    zarr_name: Optional[str] = None,
    batch_rows: int = 256,
    dry_run: bool = False,
    no_backup: bool = False,
) -> None:
    output_dir = output_dir.resolve()
    zarr_dir = (output_dir / "features") if zarr_dir is None else zarr_dir
    prepared_path = (
        output_dir / "prepared_metadata.parquet"
        if prepared_path is None
        else prepared_path
    )

    zarr_dir = zarr_dir.resolve()
    prepared_path = prepared_path.resolve()

    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared_metadata.parquet not found: {prepared_path}")
    if not zarr_dir.exists():
        raise FileNotFoundError(f"zarr_dir not found: {zarr_dir}")

    chosen_name, arr = _pick_zarr_array(zarr_dir, zarr_name)
    log.info(f"Using zarr array: {zarr_dir / chosen_name}")
    log.info(f"Zarr shape: {arr.shape}, dtype={arr.dtype}")

    log.info(f"Loading parquet: {prepared_path}")
    df = pl.read_parquet(prepared_path)

    if "zarr_index" not in df.columns:
        raise ValueError(
            "prepared_metadata.parquet is missing required column `zarr_index`"
        )
    if "complete" not in df.columns:
        raise ValueError(
            "prepared_metadata.parquet is missing required column `complete`"
        )

    n_df = len(df)
    n_zarr = int(arr.shape[0])
    if n_df != n_zarr:
        raise ValueError(
            f"Row count mismatch: parquet has {n_df:,} rows but zarr has {n_zarr:,} rows"
        )

    log.info("Computing completion mask from zarr (abs-sum > 0)")
    mask = _row_nonzero_mask(arr, batch_rows=batch_rows)

    num_from_zarr = int(mask.sum())
    num_already = int(df.filter(pl.col("complete"))["complete"].sum())

    recover_df = pl.DataFrame(
        {
            "zarr_index": np.arange(n_zarr, dtype=np.int64),
            "complete_from_zarr": pl.Series(mask),
        }
    )

    df2 = (
        df.join(recover_df, on="zarr_index", how="left")
        .with_columns(
            (pl.col("complete") | pl.col("complete_from_zarr").fill_null(False)).alias(
                "complete"
            )
        )
        .drop("complete_from_zarr")
    )

    num_after = int(df2.filter(pl.col("complete"))["complete"].sum())
    newly_completed = num_after - num_already

    log.info(
        "Completion counts: "
        f"already={num_already:,}, from_zarr={num_from_zarr:,}, after={num_after:,}, newly_marked={newly_completed:,}"
    )

    if dry_run:
        log.info("Dry-run: not writing parquet")
        return

    if not no_backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = prepared_path.with_suffix(prepared_path.suffix + f".bak-{ts}")
        shutil.copy2(prepared_path, backup_path)
        log.info(f"Wrote backup: {backup_path}")

    df2.write_parquet(prepared_path)
    log.info(f"Updated parquet written: {prepared_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover prepared_metadata.parquet `complete` flags by scanning the run's features/*.zarr "
            "and marking rows with abs-sum > 0 as complete."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run output directory (contains prepared_metadata.parquet and features/)",
    )
    parser.add_argument(
        "--zarr-dir",
        type=str,
        default=None,
        help="Optional override for zarr directory (default: <output-dir>/features)",
    )
    parser.add_argument(
        "--prepared-path",
        type=str,
        default=None,
        help="Optional override for prepared_metadata.parquet path (default: <output-dir>/prepared_metadata.parquet)",
    )
    parser.add_argument(
        "--zarr-name",
        type=str,
        default=None,
        help=(
            "Which zarr array file to scan (e.g. pred_dino.zarr). "
            f"Default: first existing among {DEFAULT_ZARR_CANDIDATES}"
        ),
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=256,
        help="How many rows to scan per read (default: 256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report counts without writing parquet",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped .bak copy of prepared_metadata.parquet",
    )

    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing runtime dependency while importing this script. "
            "Run it inside the big-img environment (where numpy/polars/zarr are installed). "
            f"Original error: {_IMPORT_ERROR}"
        )

    recover_complete_flags(
        output_dir=Path(args.output_dir),
        zarr_dir=Path(args.zarr_dir) if args.zarr_dir else None,
        prepared_path=Path(args.prepared_path) if args.prepared_path else None,
        zarr_name=args.zarr_name,
        batch_rows=int(args.batch_rows),
        dry_run=bool(args.dry_run),
        no_backup=bool(args.no_backup),
    )


if __name__ == "__main__":
    main()
