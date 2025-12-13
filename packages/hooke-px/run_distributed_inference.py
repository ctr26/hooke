"""Distributed inference master script.

Splits input parquet into chunks, launches SLURM workers to process each chunk,
monitors progress, and computes final metrics when complete.

Usage:
    python run_distributed_inference.py \
        --setup.input_parquet /path/to/observations.parquet \
        --setup.checkpoint /path/to/checkpoint.ckpt \
        --setup.output_dir /path/to/output \
        --setup.num_workers 100 \
        --setup.num_samples_per_image 1

Input parquet must have columns:
    - image_path: path to zarr image
    - cell_type: cell type string
    - experiment_label: experiment string
    - image_type: image type string
    - well_address: well address string
    - rec_id: list of compound IDs
    - concentration: list of concentrations
"""

import json
import logging
import re
import shutil
import time
from pathlib import Path

import numpy as np
import ornamentalist
import polars as pl
import submitit
import wandb
import zarr

from utils.evaluation import compute_fd, compute_cossim, compute_prdc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Feature dimensions
PHENOM_DIM = 1664
DINO_DIM = 1024
PRED_CHANNELS = 6
PRED_IMG_SIZE = 256

# Model hyperparameter keys to extract from launch_cmd.txt
MODEL_CONFIG_KEYS = ["model.name"]

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


def parse_model_config_from_checkpoint(checkpoint_path: str) -> dict | None:
    """
    Parse model hyperparameters from launch_cmd.txt in the checkpoint's grandparent directory.

    Expected structure:
        outputs/[jobnumber]/[id]/checkpoints/step_*.ckpt
        outputs/[jobnumber]/launch_cmd.txt

    Returns dict with model config or None if not found.
    """
    ckpt_path = Path(checkpoint_path)

    # Navigate: checkpoints/ -> [id]/ -> [jobnumber]/ -> launch_cmd.txt
    launch_cmd_path = ckpt_path.parent.parent.parent / "launch_cmd.txt"

    if not launch_cmd_path.exists():
        log.warning(f"launch_cmd.txt not found at {launch_cmd_path}")
        return None

    log.info(f"Found launch_cmd.txt at {launch_cmd_path}")
    launch_cmd = launch_cmd_path.read_text().strip()
    log.info(f"Launch command: {launch_cmd}")

    config = {}
    for key in MODEL_CONFIG_KEYS:
        pattern = rf"--{re.escape(key)}\s+(\S+)"
        match = re.search(pattern, launch_cmd)
        if match:
            value = match.group(1)
            short_key = key.replace("model.", "")
            if value.lower() == "true":
                config[short_key] = True
            elif value.lower() == "false":
                config[short_key] = False
            else:
                config[short_key] = value

    if config:
        log.info(f"Parsed model config: {config}")
        return config

    log.info("No model config flags found in launch_cmd.txt, using defaults")
    return None


def validate_input_parquet(df: pl.DataFrame) -> None:
    """Validate that input parquet has all required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input parquet missing required columns: {missing}. "
            f"Required: {REQUIRED_COLUMNS}"
        )


def prepare_metadata(
    input_path: str,
    split_filter: str | None = None,
    source_filter: str | None = None,
) -> pl.DataFrame:
    """
    Prepare input parquet file for inference.

    Adds required columns:
    - zarr_index: row index for zarr storage
    - complete: completion flag (default False)

    Args:
        input_path: Path to input parquet file.
        split_filter: Optional filter for 'split' column (e.g., 'val', 'test').
        source_filter: Optional filter for 'source' column (e.g., 'vcb', 'pretrain').
    """
    df = pl.read_parquet(input_path).rechunk()
    log.info(f"Loaded input parquet with {len(df)} rows")
    log.info(f"Columns: {df.columns}")

    # Validate required columns
    validate_input_parquet(df)

    # Apply filters if specified
    if split_filter is not None:
        if "split" not in df.columns:
            raise ValueError("Cannot filter by split: 'split' column not in parquet")
        df = df.filter(pl.col("split") == split_filter)
        log.info(f"Filtered to split='{split_filter}': {len(df)} rows")

    if source_filter is not None:
        if "source" not in df.columns:
            raise ValueError("Cannot filter by source: 'source' column not in parquet")
        df = df.filter(pl.col("source") == source_filter)
        log.info(f"Filtered to source='{source_filter}': {len(df)} rows")

    # Add zarr index and complete flag
    df = df.with_row_index("zarr_index").with_columns(pl.lit(False).alias("complete"))

    log.info(f"Prepared metadata with columns: {df.columns}")
    return df


def merge_worker_progress(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """Merge completion status from existing worker directories back into metadata."""
    workers_dir = output_dir / "workers"
    if not workers_dir.exists():
        return df

    existing_workers = list(workers_dir.glob("worker_*"))
    if not existing_workers:
        return df

    log.info(
        f"Found {len(existing_workers)} existing worker directories, merging progress"
    )

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


def create_worker_directories(
    df: pl.DataFrame,
    num_workers: int,
    output_dir: Path,
) -> list[Path]:
    """Split dataframe into chunks and create worker directories (only for incomplete rows)."""
    workers_dir = output_dir / "workers"

    if workers_dir.exists():
        log.info("Cleaning up old worker directories")
        shutil.rmtree(workers_dir)

    workers_dir.mkdir(parents=True, exist_ok=True)

    incomplete_df = df.filter(~pl.col("complete"))
    total_rows = len(df)
    incomplete_rows = len(incomplete_df)

    if incomplete_rows == 0:
        log.info("All rows are complete, no workers needed")
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


def create_shared_zarr(
    output_dir: Path, N: int, num_samples: int, num_real_samples: int
) -> None:
    """Initialize shared zarr arrays for feature storage (or open existing)."""
    zarr_dir = output_dir / "features"
    zarr_dir.mkdir(exist_ok=True)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    real_phenom_path = zarr_dir / "real_phenom.zarr"
    mode = "r+" if real_phenom_path.exists() else "w"

    real_phenom_shape = (
        (N, num_real_samples, PHENOM_DIM) if num_real_samples > 1 else (N, PHENOM_DIM)
    )
    real_phenom_chunks = (
        (1, num_real_samples, PHENOM_DIM) if num_real_samples > 1 else (1, PHENOM_DIM)
    )
    zarr.open(
        str(zarr_dir / "real_phenom.zarr"),
        mode=mode,
        shape=real_phenom_shape,
        chunks=real_phenom_chunks,
        dtype="float32",
        compressor=compressor,
    )
    real_dino_shape = (
        (N, num_real_samples, DINO_DIM) if num_real_samples > 1 else (N, DINO_DIM)
    )
    real_dino_chunks = (
        (1, num_real_samples, DINO_DIM) if num_real_samples > 1 else (1, DINO_DIM)
    )
    zarr.open(
        str(zarr_dir / "real_dino.zarr"),
        mode=mode,
        shape=real_dino_shape,
        chunks=real_dino_chunks,
        dtype="float32",
        compressor=compressor,
    )

    if num_samples > 1:
        zarr.open(
            str(zarr_dir / "pred_phenom.zarr"),
            mode=mode,
            shape=(N, num_samples, PHENOM_DIM),
            chunks=(1, num_samples, PHENOM_DIM),
            dtype="float32",
            compressor=compressor,
        )
        zarr.open(
            str(zarr_dir / "pred_dino.zarr"),
            mode=mode,
            shape=(N, num_samples, DINO_DIM),
            chunks=(1, num_samples, DINO_DIM),
            dtype="float32",
            compressor=compressor,
        )
        pred_images_path = zarr_dir / "pred_images.zarr"
        images_mode = "r+" if pred_images_path.exists() else "w"
        zarr.open(
            str(pred_images_path),
            mode=images_mode,
            shape=(N, num_samples, PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
            chunks=(1, num_samples, PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
            dtype="uint8",
            compressor=compressor,
        )
    else:
        zarr.open(
            str(zarr_dir / "pred_phenom.zarr"),
            mode=mode,
            shape=(N, PHENOM_DIM),
            chunks=(1, PHENOM_DIM),
            dtype="float32",
            compressor=compressor,
        )
        pred_images_path = zarr_dir / "pred_images.zarr"
        images_mode = "r+" if pred_images_path.exists() else "w"
        zarr.open(
            str(pred_images_path),
            mode=images_mode,
            shape=(N, PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
            chunks=(1, PRED_CHANNELS, PRED_IMG_SIZE, PRED_IMG_SIZE),
            dtype="uint8",
            compressor=compressor,
        )
        zarr.open(
            str(zarr_dir / "pred_dino.zarr"),
            mode=mode,
            shape=(N, DINO_DIM),
            chunks=(1, DINO_DIM),
            dtype="float32",
            compressor=compressor,
        )

    if mode == "r+":
        log.info(f"Opened existing zarr arrays at {zarr_dir}")
    else:
        log.info(f"Created new zarr arrays at {zarr_dir}")


def run_worker_job(worker_dir: str, config_path: str) -> str:
    """Worker function that submitit will execute on each SLURM node."""
    from run_inference_worker import run_worker

    run_worker(worker_dir, config_path)
    return worker_dir


def launch_workers(
    worker_dirs: list[Path],
    config_path: Path,
    output_dir: Path,
    partition: str = "hopper",
    qos: str | None = "hooke-predict",
    timeout_min: int = 120,
) -> list[submitit.Job]:
    """Launch SLURM jobs for each worker directory using submitit array job."""
    executor = submitit.AutoExecutor(folder=str(output_dir / "submitit_logs"))

    executor.update_parameters(
        slurm_partition=partition,
        slurm_qos=qos,
        slurm_wckey="hooke-predict",
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=6,
        mem_gb=32,
        timeout_min=timeout_min,
        slurm_additional_parameters={"requeue": True},
    )

    worker_args = [(str(wd), str(config_path)) for wd in worker_dirs]
    jobs = executor.map_array(run_worker_job, *zip(*worker_args))

    log.info(f"Submitted array job with {len(jobs)} workers")
    if jobs:
        log.info(f"Job IDs: {jobs[0].job_id} (array of {len(jobs)})")

    return jobs


def monitor_progress(
    jobs: list[submitit.Job],
    worker_dirs: list[Path],
    check_interval: int = 60,
    use_wandb: bool = False,
    wandb_log_fn=None,
) -> None:
    """Monitor submitit jobs until all complete."""
    total_workers = len(jobs)
    iteration = 0

    while True:
        iteration += 1
        time.sleep(check_interval)

        completed = sum(1 for j in jobs if j.done())
        running = sum(1 for j in jobs if j.state == "RUNNING")
        pending = sum(1 for j in jobs if j.state == "PENDING")
        failed = sum(1 for j in jobs if j.state == "FAILED")

        dirs_completed = 0
        for worker_dir in worker_dirs:
            if worker_dir.exists():
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

        if use_wandb and wandb_log_fn:
            wandb_log_fn(
                {
                    "inference/completed_workers": dirs_completed,
                    "inference/total_workers": total_workers,
                    "inference/running_jobs": running,
                    "inference/pending_jobs": pending,
                    "inference/failed_jobs": failed,
                },
                step=iteration,
            )

        if dirs_completed == total_workers:
            break

        if completed == total_workers:
            incomplete_workers = total_workers - dirs_completed
            if incomplete_workers > 0:
                log.warning(
                    f"{incomplete_workers} workers still incomplete after all jobs completed"
                )
            break

    log.info("All workers complete!")


def compute_final_metrics(
    output_dir: Path, num_samples: int, num_real_samples: int
) -> dict:
    """Load features from zarr and compute metrics."""
    zarr_dir = output_dir / "features"

    real_phenom = np.array(zarr.open(str(zarr_dir / "real_phenom.zarr"), mode="r"))
    real_dino = np.array(zarr.open(str(zarr_dir / "real_dino.zarr"), mode="r"))
    pred_phenom = np.array(zarr.open(str(zarr_dir / "pred_phenom.zarr"), mode="r"))
    pred_dino = np.array(zarr.open(str(zarr_dir / "pred_dino.zarr"), mode="r"))

    N = len(real_phenom)
    log.info(f"Computing metrics for {N} samples")

    if num_samples > 1:
        pred_phenom_mean = pred_phenom.mean(axis=1)
        pred_phenom_samples = pred_phenom.reshape(-1, PHENOM_DIM)
        pred_dino_mean = pred_dino.mean(axis=1)
        pred_dino_samples = pred_dino.reshape(-1, DINO_DIM)
    else:
        pred_phenom_mean = pred_phenom_samples = pred_phenom.reshape(-1, PHENOM_DIM)
        pred_dino_mean = pred_dino_samples = pred_dino.reshape(-1, DINO_DIM)

    if real_phenom.ndim == 3 and num_real_samples > 1:
        real_phenom_mean = real_phenom.mean(axis=1)
        real_phenom_samples = real_phenom.reshape(-1, PHENOM_DIM)
        real_dino_mean = real_dino.mean(axis=1)
        real_dino_samples = real_dino.reshape(-1, DINO_DIM)
    else:
        real_phenom_mean = real_phenom_samples = real_phenom.reshape(-1, PHENOM_DIM)
        real_dino_mean = real_dino_samples = real_dino.reshape(-1, DINO_DIM)

    metrics = {}

    metrics["fd_phenom"] = compute_fd(real_phenom_samples, pred_phenom_samples)
    metrics["fd_dino"] = compute_fd(real_dino_samples, pred_dino_samples)

    metrics["cossim_phenom"] = compute_cossim(real_phenom_mean, pred_phenom_mean)
    metrics["cossim_dino"] = compute_cossim(real_dino_mean, pred_dino_mean)

    SUBSAMPLE_N = 10_000
    real_total = real_dino_samples.shape[0]
    pred_total = pred_dino_samples.shape[0]
    paired_total = min(real_total, pred_total)
    if paired_total > SUBSAMPLE_N:
        rng = np.random.RandomState(seed=42)
        idxs = rng.choice(paired_total, size=SUBSAMPLE_N, replace=False)
        real_dino_sub = real_dino_samples[idxs]
        pred_dino_sub = pred_dino_samples[idxs]
    else:
        real_dino_sub = real_dino_samples[:paired_total]
        pred_dino_sub = pred_dino_samples[:paired_total]

    prdc = compute_prdc(real_dino_sub, pred_dino_sub, nearest_k=5)
    for k, v in prdc.items():
        metrics[f"{k}_dino"] = v

    log.info(f"Metrics: {metrics}")
    return metrics


@ornamentalist.configure(name="setup")
def distributed_inference(
    input_parquet: str = ornamentalist.Configurable[""],
    checkpoint: str = ornamentalist.Configurable[""],
    output_dir: str = ornamentalist.Configurable[""],
    num_workers: int = ornamentalist.Configurable[100],
    num_samples_per_image: int = ornamentalist.Configurable[1],
    num_real_image_samples: int = ornamentalist.Configurable[1],
    batch_size: int = ornamentalist.Configurable[4],
    partition: str = ornamentalist.Configurable["hopper"],
    qos: str = ornamentalist.Configurable["hooke-predict"],
    wandb_project: str | None = ornamentalist.Configurable[None],
    skip_launch: bool = ornamentalist.Configurable[False],
    metrics_only: bool = ornamentalist.Configurable[False],
    split: str | None = ornamentalist.Configurable[None],
    source: str | None = ornamentalist.Configurable[None],
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=output_dir.parent.name + "_" + output_dir.name,
            config=ornamentalist.get_config(),
        )

    num_real_image_samples = max(1, int(num_real_image_samples))

    if metrics_only:
        metrics = compute_final_metrics(
            output_dir, num_samples_per_image, num_real_image_samples
        )
        if use_wandb:
            wandb.log(metrics)
            wandb.finish()
        return

    # Parse model config from checkpoint's launch_cmd.txt
    model_config = parse_model_config_from_checkpoint(checkpoint)

    # Check if we're resuming from existing run
    prepared_path = output_dir / "prepared_metadata.parquet"
    if prepared_path.exists():
        log.info(f"Resuming from existing run: {prepared_path}")
        df = pl.read_parquet(prepared_path)
        N = len(df)

        df = merge_worker_progress(df, output_dir)

        num_complete = df.filter(pl.col("complete"))["complete"].sum()
        log.info(f"Found {num_complete}/{N} already complete")

        df.write_parquet(prepared_path)
    else:
        df = prepare_metadata(
            input_parquet,
            split_filter=split,
            source_filter=source,
        )
        N = len(df)
        df.write_parquet(prepared_path)
        log.info(f"Prepared {N} rows, saved to {prepared_path}")

    # Create config for workers (for ornamentalist injection)
    worker_conf = {
        "get_worker_config": {
            "checkpoint_path": checkpoint,
            "zarr_dir": str(output_dir / "features"),
            "num_samples_per_image": num_samples_per_image,
            "num_real_image_samples": num_real_image_samples,
            "batch_size": batch_size,
            "num_workers": 4,
        }
    }

    if model_config:
        worker_conf["get_model_cls"] = model_config

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(worker_conf, f, indent=2)

    create_shared_zarr(output_dir, N, num_samples_per_image, num_real_image_samples)

    worker_dirs = create_worker_directories(df, num_workers, output_dir)

    if len(worker_dirs) == 0:
        log.info("All work already complete, skipping to metrics")
    elif skip_launch:
        log.info("Skipping worker launch (--skip_launch)")
        return
    else:
        log.info(f"Launching {len(worker_dirs)} workers")
        actual_qos = qos if qos != "default" else None

        jobs = launch_workers(
            worker_dirs, config_path, output_dir, partition, actual_qos
        )

        monitor_progress(
            jobs,
            worker_dirs,
            use_wandb=use_wandb,
            wandb_log_fn=wandb.log if use_wandb else None,
        )

        workers_dir = output_dir / "workers"
        if workers_dir.exists():
            log.info("Cleaning up worker directories")
            shutil.rmtree(workers_dir)

    metrics = compute_final_metrics(
        output_dir, num_samples_per_image, num_real_image_samples
    )

    if use_wandb:
        wandb.log(metrics)
        wandb.finish()

    log.info("Distributed inference complete!")


if __name__ == "__main__":
    configs = ornamentalist.cli()
    ornamentalist.setup(configs[0], force=True)
    distributed_inference()
