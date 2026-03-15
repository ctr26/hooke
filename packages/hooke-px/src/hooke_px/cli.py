#!/usr/bin/env python
"""Hydra CLI for hooke-px pipeline.

Usage:
    # Run inference only
    python -m hooke_px.cli inference checkpoint=hooke-px/pretrain-checkpoint:latest

    # Run full pipeline (inference + eval)
    python -m hooke_px.cli pipeline

    # Override config
    python -m hooke_px.cli inference batch_size=6 partition=gpu
"""

import hydra
import weave
from omegaconf import DictConfig

from hooke_px.configs import PipelineConfig, register_configs
from hooke_px.inference.step import inference_step, submit_inference
from hooke_px.schemas import EvalInput, InferenceInput

# Register configs
register_configs()


@hydra.main(version_base=None, config_name="pipeline")
def main(cfg: DictConfig) -> None:
    """Run pipeline with Hydra config."""
    weave.init(cfg.weave_project)

    # Build inference input from config
    inference_input = InferenceInput(
        checkpoint_path=cfg.inference.checkpoint,
        dataset_path=cfg.inference.dataset,
        output_dir=cfg.inference.output_dir,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        num_samples=cfg.inference.num_samples,
        partition=cfg.inference.partition,
        gpus_per_node=cfg.inference.gpus_per_node,
    )

    if cfg.get("run_eval", False):
        # Full pipeline: inference + eval
        from hooke_px.pipeline import run_pipeline

        result = run_pipeline(
            checkpoint_path=cfg.inference.checkpoint,
            dataset_path=cfg.inference.dataset,
            output_dir=cfg.inference.output_dir,
            ground_truth_path=cfg.eval.ground_truth_path,
            split_path=cfg.eval.split_path,
            batch_size=cfg.inference.batch_size,
            num_workers=cfg.inference.num_workers,
            num_samples=cfg.inference.num_samples,
            partition=cfg.inference.partition,
            gpus_per_node=cfg.inference.gpus_per_node,
            weave_project=cfg.weave_project,
        )
        print(f"Pipeline complete: {result['metrics']}")
    else:
        # Inference only — submit to SLURM
        job = submit_inference(inference_input)
        print(f"Job submitted: {job.job_id}")


if __name__ == "__main__":
    main()
