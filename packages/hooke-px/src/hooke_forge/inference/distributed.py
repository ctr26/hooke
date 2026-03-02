"""Distributed inference orchestration.

Manages SLURM worker jobs for parallel inference across a dataset.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import polars as pl
import submitit
import zarr
import numcodecs

from hooke_forge.inference.run_worker import (
    REPRESENTATION_DIMS,
    DEFAULT_PX_REPRESENTATIONS,
    DEFAULT_TX_REPRESENTATIONS,
    TX_ASSAY_TYPES,
    detect_modality,
)

log = logging.getLogger(__name__)

# Common metadata columns required for all modalities
_COMMON_COLUMNS = [
    "cell_type",
    "experiment_label",
    "well_address",
    "rec_id",
    "concentration",
]

# Additional columns per modality
_PX_COLUMNS = ["image_path", "image_type"]
_TX_COLUMNS = ["zarr_row_idx"]


def _get_required_columns(modality: str) -> list[str]:
    extra = _TX_COLUMNS if modality == "tx" else _PX_COLUMNS
    return _COMMON_COLUMNS + extra


def prepare_metadata(
    input_parquet: Path,
    output_dir: Path,
    modality: str,
    split_filter: str | None = None,
    source_filter: str | None = None,
) -> pl.DataFrame:
    """Prepare input parquet for inference.

    Adds zarr_index and complete columns. Optionally filters by split/source.
    """
    df = pl.read_parquet(input_parquet).rechunk()
    log.info(f"Loaded input parquet: {len(df)} rows")

    # Normalise assay_type / image_type
    if "assay_type" not in df.columns and "image_type" in df.columns:
        df = df.with_columns(pl.col("image_type").alias("assay_type"))

    # Validate required columns
    required = _get_required_columns(modality)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {modality} modality: {missing}")

    # Apply filters
    if split_filter is not None and "split" in df.columns:
        df = df.filter(pl.col("split") == split_filter)
        log.info(f"Filtered to split='{split_filter}': {len(df)} rows")

    if source_filter is not None and "source" in df.columns:
        df = df.filter(pl.col("source") == source_filter)
        log.info(f"Filtered to source='{source_filter}': {len(df)} rows")

    # Add zarr index and complete flag
    df = df.with_row_index("zarr_index").with_columns(pl.lit(False).alias("complete"))

    # Save
    prepared_path = output_dir / "prepared_metadata.parquet"
    df.write_parquet(prepared_path)
    log.info(f"Saved prepared metadata: {prepared_path}")

    return df


def merge_worker_progress(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Merge completion status from existing worker directories."""
    workers_dir = output_dir / "workers"
    if not workers_dir.exists():
        return df

    existing_workers = list(workers_dir.glob("worker_*"))
    if not existing_workers:
        return df

    log.info(f"Found {len(existing_workers)} existing worker directories")

    completed_indices = set()
    for worker_dir in existing_workers:
        chunk_path = worker_dir / "chunk.parquet"
        if chunk_path.exists():
            chunk_df = pl.read_parquet(chunk_path)
            completed = chunk_df.filter(pl.col("complete"))
            if len(completed) > 0:
                completed_indices.update(completed["zarr_index"].to_list())

    if completed_indices:
        log.info(f"Found {len(completed_indices)} completed rows from existing workers")
        df = df.with_columns(
            pl.when(pl.col("zarr_index").is_in(list(completed_indices)))
            .then(pl.lit(True))
            .otherwise(pl.col("complete"))
            .alias("complete")
        )

    return df


def create_zarr_arrays(
    output_dir: Path,
    num_obs: int,
    num_samples: int,
    representations: list[str],
    tx_feature_dim: int | None = None,
) -> None:
    """Initialize shared zarr arrays for the requested representations only."""
    zarr_dir = output_dir / "features"
    zarr_dir.mkdir(exist_ok=True)

    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)

    if num_samples > 1:
        sample_shape = (num_obs, num_samples)
        sample_chunks = (1, num_samples)
    else:
        sample_shape = (num_obs,)
        sample_chunks = (1,)

    for name in representations:
        dim, dtype = REPRESENTATION_DIMS[name]

        # pred_tx has runtime-determined dimension
        if dim is None:
            if tx_feature_dim is None:
                raise ValueError(f"tx_feature_dim must be provided for {name}")
            dim = (tx_feature_dim,)

        # real_* features are never multi-sampled
        if name.startswith("real_"):
            shape = (num_obs,) + dim
            chunks = (1,) + dim
        else:
            shape = sample_shape + dim
            chunks = sample_chunks + dim

        zarr_path = zarr_dir / f"{name}.zarr"
        mode = "r+" if zarr_path.exists() else "w"
        zarr.open(
            str(zarr_path),
            mode=mode,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            zarr_format=2,
        )
        log.info(f"{'Opened' if mode == 'r+' else 'Created'} {name}.zarr shape={shape}")


def create_worker_directories(
    df: pl.DataFrame,
    num_workers: int,
    output_dir: Path,
) -> list[Path]:
    """Split incomplete rows into worker directories."""
    workers_dir = output_dir / "workers"

    if workers_dir.exists():
        shutil.rmtree(workers_dir)
    workers_dir.mkdir(parents=True, exist_ok=True)

    incomplete_df = df.filter(~pl.col("complete"))
    total_rows = len(df)
    incomplete_rows = len(incomplete_df)

    if incomplete_rows == 0:
        log.info("All rows complete, no workers needed")
        return []

    log.info(f"Creating workers for {incomplete_rows}/{total_rows} incomplete rows")

    chunk_size = (incomplete_rows + num_workers - 1) // num_workers
    worker_dirs = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, incomplete_rows)
        if start_idx >= incomplete_rows:
            break

        chunk = incomplete_df.slice(start_idx, end_idx - start_idx)
        worker_dir = workers_dir / f"worker_{i}"
        worker_dir.mkdir(exist_ok=True)
        chunk.write_parquet(worker_dir / "chunk.parquet")
        worker_dirs.append(worker_dir)

    log.info(f"Created {len(worker_dirs)} worker directories")
    return worker_dirs


def _run_worker_job(worker_dir: str, config_path: str) -> str:
    """Worker function executed on each SLURM node."""
    from hooke_forge.inference.run_worker import run_worker

    run_worker(worker_dir, config_path)
    return worker_dir


def launch_workers(
    worker_dirs: list[Path],
    config_path: Path,
    output_dir: Path,
    partition: str = "hopper",
    qos: str | None = None,
    timeout_min: int = 480,
) -> list[submitit.Job]:
    """Launch SLURM array job for workers."""
    executor = submitit.AutoExecutor(folder=str(output_dir / "submitit_logs"))

    executor.update_parameters(
        slurm_partition=partition,
        slurm_qos=qos,
        slurm_wckey="hooke-predict",
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=6,
        mem_gb=48,
        timeout_min=timeout_min,
        slurm_additional_parameters={
            "requeue": True,
            "exclude": "hop01,hop61,hop62",
        },
    )

    worker_args = [(str(wd), str(config_path)) for wd in worker_dirs]
    jobs = executor.map_array(_run_worker_job, *zip(*worker_args))

    log.info(f"Submitted array job with {len(jobs)} workers")
    if jobs:
        log.info(f"Job IDs: {jobs[0].job_id} (array of {len(jobs)})")

    return jobs


def monitor_until_complete(
    jobs: list[submitit.Job],
    worker_dirs: list[Path],
    check_interval: int = 60,
) -> bool:
    """Monitor jobs until all complete.

    Returns:
        True if all completed successfully, False otherwise
    """
    total_workers = len(jobs)

    while True:
        time.sleep(check_interval)

        completed = sum(1 for j in jobs if j.done())
        running = sum(1 for j in jobs if j.state == "RUNNING")
        pending = sum(1 for j in jobs if j.state == "PENDING")
        failed = sum(1 for j in jobs if j.state == "FAILED")

        dirs_completed = 0
        for worker_dir in worker_dirs:
            chunk_path = worker_dir / "chunk.parquet"
            if chunk_path.exists():
                try:
                    chunk_df = pl.read_parquet(chunk_path)
                    if chunk_df["complete"].all():
                        dirs_completed += 1
                except Exception:
                    pass

        log.info(
            f"Progress: {dirs_completed}/{total_workers} dirs complete, "
            f"jobs: {completed} done, {running} running, {pending} pending, {failed} failed"
        )

        if dirs_completed == total_workers:
            break

        if completed == total_workers:
            if dirs_completed < total_workers:
                log.warning(f"{total_workers - dirs_completed} workers incomplete after all jobs done")
            break

    log.info("All workers complete!")
    return dirs_completed == total_workers


def run_distributed_inference(
    checkpoint_path: Path,
    input_parquet: Path,
    output_dir: Path,
    model_config: dict | None = None,
    flow_model_config: dict | None = None,
    num_workers: int = 100,
    num_samples: int = 36,
    num_real_samples: int = 1,
    batch_size: int = 3,
    partition: str = "hopper",
    qos: str | None = None,
    representations: list[str] | None = None,
    tx_zarr_path: str = "",
) -> Path:
    """Run distributed inference pipeline.

    Args:
        checkpoint_path: Path to model checkpoint
        input_parquet: Path to input metadata parquet
        output_dir: Output directory
        model_config: Optional model config dict (e.g. {"name": "DiT-XL/2"})
        flow_model_config: Optional flow_model config dict (e.g. {"modality": "tx"})
        num_workers: Number of SLURM workers
        num_samples: Samples per observation for predictions
        num_real_samples: Samples for real image features
        batch_size: Batch size per worker
        partition: SLURM partition
        qos: SLURM QOS
        representations: List of representations to extract (auto-detected if None)
        tx_zarr_path: Path to tx feature zarr (required for tx modality)

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running distributed inference")
    log.info(f"  Checkpoint: {checkpoint_path}")
    log.info(f"  Input: {input_parquet}")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Workers: {num_workers}")

    # Auto-detect modality from input parquet
    peek_df = pl.read_parquet(input_parquet, n_rows=1)
    assay_col = "assay_type" if "assay_type" in peek_df.columns else "image_type"
    assay_type = peek_df[assay_col][0]
    modality = detect_modality(assay_type)
    log.info(f"  Modality: {modality} (assay_type={assay_type})")

    # Default representations
    if representations is None:
        representations = DEFAULT_TX_REPRESENTATIONS if modality == "tx" else DEFAULT_PX_REPRESENTATIONS
    log.info(f"  Representations: {representations}")

    # Check for resume
    prepared_path = output_dir / "prepared_metadata.parquet"
    if prepared_path.exists():
        log.info("Resuming from existing run")
        df = pl.read_parquet(prepared_path)
        df = merge_worker_progress(df, output_dir)
        num_complete = df.filter(pl.col("complete"))["complete"].sum()
        log.info(f"Found {num_complete}/{len(df)} already complete")
        df.write_parquet(prepared_path)
    else:
        df = prepare_metadata(input_parquet, output_dir, modality=modality)

    N = len(df)

    # Determine tx_feature_dim if needed
    tx_feature_dim = None
    if "pred_tx" in representations:
        fm_cfg = flow_model_config or {}
        tx_feature_dim = fm_cfg.get("tx_feature_dim", 5000)

    # Create worker config
    worker_conf = {
        "get_worker_config": {
            "checkpoint_path": str(checkpoint_path),
            "zarr_dir": str(output_dir / "features"),
            "num_samples_per_image": num_samples,
            "num_real_image_samples": num_real_samples,
            "batch_size": batch_size,
            "num_workers": 4,
            "representations": representations,
            "tx_zarr_path": tx_zarr_path,
        }
    }
    if model_config:
        worker_conf["model"] = model_config
    if flow_model_config:
        worker_conf["flow_model"] = flow_model_config

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(worker_conf, f, indent=2)

    # Create zarr arrays for requested representations only
    create_zarr_arrays(output_dir, N, num_samples, representations, tx_feature_dim)

    # Create worker directories
    worker_dirs = create_worker_directories(df, num_workers, output_dir)

    if len(worker_dirs) == 0:
        log.info("All work already complete")
        return output_dir

    # Launch workers
    jobs = launch_workers(worker_dirs, config_path, output_dir, partition, qos)

    # Monitor until complete
    monitor_until_complete(jobs, worker_dirs)

    # Cleanup
    workers_dir = output_dir / "workers"
    if workers_dir.exists():
        log.info("Cleaning up worker directories")
        shutil.rmtree(workers_dir)

    log.info("Distributed inference complete!")
    return output_dir
