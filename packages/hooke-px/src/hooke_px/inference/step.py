"""Inference step with Weave lineage + Submitit + joblib caching.

Weave: tracks inputs/outputs/lineage
Submitit: submits GPU jobs to SLURM
joblib.Memory: skips re-computation if already done
"""

import logging
from pathlib import Path

import submitit
import weave
from joblib import Memory

from hooke_px.schemas import InferenceInput, InferenceOutput

log = logging.getLogger(__name__)

# joblib cache at shared location
memory = Memory("/data/valence/cache", verbose=0)


def _resolve_checkpoint_path(checkpoint_ref: str) -> str:
    """Resolve checkpoint from W&B artifact ref, Weave ref, or local path.

    Supports:
    - Local path: /path/to/checkpoint.ckpt → passthrough
    - W&B artifact: entity/project/artifact:version → download and return path
    - Weave ref: hooke-px/pretrain-checkpoint:latest → resolve via weave
    """
    path = Path(checkpoint_ref)

    # Local path — passthrough
    if path.exists():
        return str(path)

    # Weave ref (starts with project name pattern)
    if "/" in checkpoint_ref and ":" in checkpoint_ref:
        try:
            artifact = weave.ref(checkpoint_ref).get()
            if hasattr(artifact, "path"):
                return str(artifact.path)
            # W&B artifact downloaded — find .ckpt in the directory
            if hasattr(artifact, "download"):
                artifact_dir = Path(artifact.download())
                ckpts = list(artifact_dir.rglob("*.ckpt")) + list(artifact_dir.rglob("*.pt"))
                if ckpts:
                    return str(ckpts[0])
                return str(artifact_dir)
        except Exception:
            log.debug(f"Weave ref resolution failed for {checkpoint_ref}, trying W&B API")

        # Try W&B artifact API
        try:
            import wandb

            api = wandb.Api()
            artifact = api.artifact(checkpoint_ref)
            artifact_dir = Path(artifact.download())
            ckpts = list(artifact_dir.rglob("*.ckpt")) + list(artifact_dir.rglob("*.pt"))
            if ckpts:
                return str(ckpts[0])
            return str(artifact_dir)
        except Exception:
            log.debug(f"W&B artifact resolution failed for {checkpoint_ref}")

    # If nothing worked, return as-is (let downstream handle the error)
    return checkpoint_ref


@memory.cache
def _run_inference_cached(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    num_samples: int,
) -> str:
    """Cached inference execution.

    joblib hashes all args — same inputs = skip.
    Returns path to features.
    """
    from hooke_px.inference.run import run_inference_job

    features_path = run_inference_job(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
    )

    return features_path


@weave.op()
def inference_step(input: InferenceInput) -> InferenceOutput:
    """Run inference with Weave lineage + joblib caching.

    Weave tracks:
    - Input schema (checkpoint ref, dataset, config)
    - Output schema (features path, sample count)
    - Checkpoint reference for lineage tracing
    """
    checkpoint_path = _resolve_checkpoint_path(input.checkpoint_path)
    log.info(f"Resolved checkpoint: {checkpoint_path}")

    # Run (or skip via joblib cache)
    features_path = _run_inference_cached(
        checkpoint_path=checkpoint_path,
        dataset_path=input.dataset_path,
        output_dir=input.output_dir,
        batch_size=input.batch_size,
        num_workers=input.num_workers,
        num_samples=input.num_samples,
    )

    # Count samples from zarr
    num_samples = 0
    features_dir = Path(features_path)
    if features_dir.is_dir():
        import zarr

        for zarr_path in features_dir.glob("*.zarr"):
            arr = zarr.open(str(zarr_path), "r")
            if hasattr(arr, "shape"):
                num_samples = arr.shape[0]
                break
    elif features_dir.exists():
        import zarr

        arr = zarr.open(str(features_dir), "r")
        if hasattr(arr, "shape"):
            num_samples = arr.shape[0]

    return InferenceOutput(
        features_path=features_path,
        num_samples=num_samples,
        checkpoint_ref=input.checkpoint_path,
    )


def submit_inference(input: InferenceInput) -> submitit.Job:
    """Submit inference as SLURM job via Submitit."""
    executor = submitit.AutoExecutor(folder="logs/inference")
    executor.update_parameters(
        slurm_partition=input.partition,
        gpus_per_node=input.gpus_per_node,
        slurm_time="24:00:00",
        slurm_mem="128G",
    )

    job = executor.submit(inference_step, input)
    log.info(f"Submitted inference job: {job.job_id}")

    return job
