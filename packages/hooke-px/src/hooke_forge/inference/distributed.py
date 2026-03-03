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

log = logging.getLogger(__name__)

# Feature dimensions
PHENOM_DIM = 1664
DINO_DIM = 1024
PRED_CHANNELS = 6
PRED_IMG_SIZE = 256

# Required columns in input parquet
REQUIRED_COLUMNS = [
    "image_path",
    "cell_type",
    "experiment_label",
    "image_type",
    "well_address",
    "rec_id",
    "concentration",
]


def prepare_metadata(
    input_parquet: Path,
    output_dir: Path,
    split_filter: str | None = None,
    source_filter: str | None = None,
    vcb_dataset: str | None = None,
) -> pl.DataFrame:
    """Prepare input parquet for inference.

    Adds zarr_index and complete columns. Optionally filters by split/source.

    Args:
        input_parquet: Path to input parquet file
        output_dir: Output directory (saves prepared_metadata.parquet)
        split_filter: Optional filter for split column
        source_filter: Optional filter for source column
        vcb_dataset: Optional VCB dataset type to transform obs parquet

    Returns:
        Prepared DataFrame
    """
    df = pl.read_parquet(input_parquet).rechunk()
    log.info(f"Loaded input parquet: {len(df)} rows")

    # Apply VCB transformation if specified
    if vcb_dataset:
        from hooke_forge.inference.vcb_datasets import transform_vcb_dataset

        df = transform_vcb_dataset(df, vcb_dataset)
        log.info(f"Applied VCB transformation for '{vcb_dataset}'")

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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
    num_real_samples: int,
) -> None:
    """Initialize shared zarr arrays for feature storage."""
    zarr_dir = output_dir / "features"
    zarr_dir.mkdir(exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    real_phenom_path = zarr_dir / "real_phenom.zarr"
    mode = "r+" if real_phenom_path.exists() else "w"

    # Real features
    if num_real_samples > 1:
        real_shape = (num_obs, num_real_samples)
        real_chunks = (1, num_real_samples)
    else:
        real_shape = (num_obs,)
        real_chunks = (1,)

    zarr.open(
        str(zarr_dir / "real_phenom.zarr"),
        mode=mode,
        shape=real_shape + (PHENOM_DIM,),
        chunks=real_chunks + (PHENOM_DIM,),
        dtype="float32",
        compressor=compressor,
    )
    zarr.open(
        str(zarr_dir / "real_dino.zarr"),
        mode=mode,
        shape=real_shape + (DINO_DIM,),
        chunks=real_chunks + (DINO_DIM,),
        dtype="float32",
        compressor=compressor,
    )

    # Predicted features
    if num_samples > 1:
        pred_shape = (num_obs, num_samples)
        pred_chunks = (1, num_samples)
    else:
        pred_shape = (num_obs,)
        pred_chunks = (1,)

    zarr.open(
        str(zarr_dir / "pred_phenom.zarr"),
        mode=mode,
        shape=pred_shape + (PHENOM_DIM,),
        chunks=pred_chunks + (PHENOM_DIM,),
        dtype="float32",
        compressor=compressor,
    )
    zarr.open(
        str(zarr_dir / "pred_dino.zarr"),
        mode=mode,
        shape=pred_shape + (DINO_DIM,),
        chunks=pred_chunks + (DINO_DIM,),
        dtype="float32",
        compressor=compressor,
    )

    # Predicted images
    pred_images_path = zarr_dir / "pred_images.zarr"
    images_mode = "r+" if pred_images_path.exists() else "w"
    zarr.open(
        str(pred_images_path),
        mode=images_mode,
        shape=pred_shape + (PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
        chunks=pred_chunks + (PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
        dtype="uint8",
        compressor=compressor,
    )

    log.info(f"{'Opened existing' if mode == 'r+' else 'Created'} zarr arrays at {zarr_dir}")


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
    num_workers: int = 100,
    num_samples: int = 36,
    num_real_samples: int = 1,
    batch_size: int = 3,
    partition: str = "hopper",
    qos: str | None = None,
    vcb_dataset: str | None = None,
) -> Path:
    """Run distributed inference pipeline.

    Args:
        checkpoint_path: Path to model checkpoint
        input_parquet: Path to input metadata parquet
        output_dir: Output directory
        model_config: Optional model config dict
        num_workers: Number of SLURM workers
        num_samples: Samples per image for predictions
        num_real_samples: Samples for real image features
        batch_size: Batch size per worker
        partition: SLURM partition
        qos: SLURM QOS
        vcb_dataset: Optional VCB dataset type to transform obs parquet

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
        df = prepare_metadata(input_parquet, output_dir, vcb_dataset=vcb_dataset)

    N = len(df)

    # Create worker config
    worker_conf = {
        "get_worker_config": {
            "checkpoint_path": str(checkpoint_path),
            "zarr_dir": str(output_dir / "features"),
            "num_samples_per_image": num_samples,
            "num_real_image_samples": num_real_samples,
            "batch_size": batch_size,
            "num_workers": 4,
        }
    }
    if model_config:
        worker_conf["model"] = model_config

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(worker_conf, f, indent=2)

    # Create zarr arrays
    create_zarr_arrays(output_dir, N, num_samples, num_real_samples)

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
