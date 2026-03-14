"""Inference step with Weave lineage + Submitit + joblib caching.

Weave: tracks inputs/outputs/lineage
Submitit: submits GPU jobs to SLURM
joblib.Memory: skips re-computation if already done
"""

from pathlib import Path

import submitit
import weave
from joblib import Memory

from hooke_px.schemas import InferenceInput, InferenceOutput

# joblib cache at shared location
memory = Memory("/data/valence/cache", verbose=0)


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
    - Input schema
    - Output schema
    - Checkpoint reference (lineage)

    joblib skips if already computed with same inputs.
    """
    # Resolve checkpoint from W&B if needed
    if input.checkpoint_path.startswith("hooke-"):
        # It's a weave ref
        checkpoint_artifact = weave.ref(input.checkpoint_path).get()
        checkpoint_path = str(checkpoint_artifact.path)
    else:
        checkpoint_path = input.checkpoint_path

    # Run (or skip via cache)
    features_path = _run_inference_cached(
        checkpoint_path=checkpoint_path,
        dataset_path=input.dataset_path,
        output_dir=input.output_dir,
        batch_size=input.batch_size,
        num_workers=input.num_workers,
        num_samples=input.num_samples,
    )

    # Count samples
    import zarr
    features = zarr.open(features_path, "r")
    num_samples = features.shape[0] if hasattr(features, "shape") else 0

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
    print(f"Submitted inference job: {job.job_id}")

    return job
