#!/usr/bin/env python
"""Schema-governed pipeline demo — builder/accumulator pattern.

A single PipelineState accumulates fields at each step.
Steps add to it, never flatten or nest.

    uv run python demo/schema_pipeline_accumulator.py
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


# -- Single accumulator: grows as the pipeline progresses --


class PipelineState(BaseModel):
    # Data (set by splits_step)
    split_file: str = "data/splits/default.json"
    split_path: str | None = None
    train_compounds: list[str] | None = None
    val_compounds: list[str] | None = None
    test_compounds: list[str] | None = None

    # Conditioning (set by conditioning_step)
    cell_types: list[str] | None = None
    assay_types: list[str] | None = None
    vocab_size: int | None = None
    conditioning_path: str | None = None

    # Pretrain (set by pretrain_step)
    pretrain_checkpoint: str | None = None
    pretrain_step_count: int | None = None

    # Finetuning (set by finetuning_step)
    finetune_checkpoint: str | None = None
    finetune_step_count: int | None = None
    target_cell_type: str | None = None

    # Inference (set by inference_step)
    features_path: str | None = None
    num_samples: int | None = None

    # Eval (set by eval_step)
    metrics: dict[str, float] | None = None


# -- Steps: each enriches the state --


@weave.op()
def splits_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "split_path": state.split_file,
        "train_compounds": ["cpd_001", "cpd_002", "cpd_003"],
        "val_compounds": ["cpd_004"],
        "test_compounds": ["cpd_005", "cpd_006"],
    })


@weave.op()
def conditioning_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "cell_types": ["ARPE19", "HUVEC", "HepG2"],
        "assay_types": ["cell_paint", "brightfield"],
        "vocab_size": 2048,
        "conditioning_path": "/data/conditioning/v1.json",
    })


@weave.op()
def pretrain_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "pretrain_checkpoint": "/checkpoints/pretrain_200k.pt",
        "pretrain_step_count": 200_000,
    })


@weave.op()
def finetuning_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "finetune_checkpoint": f"{state.pretrain_checkpoint}.finetuned",
        "finetune_step_count": (state.pretrain_step_count or 0) + 50_000,
        "target_cell_type": "ARPE19",
    })


@weave.op()
def inference_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "features_path": "/outputs/features.npy",
        "num_samples": len(state.test_compounds or []) * 100,
    })


@weave.op()
def eval_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "metrics": {"map_cosine": 0.85, "pearson": 0.72},
    })


# -- Pipeline --


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo-accumulator"


def run_pipeline(cfg: PipelineConfig) -> PipelineState:
    weave.init(cfg.project)

    state = PipelineState()
    steps = [splits_step, conditioning_step, pretrain_step, finetuning_step, inference_step, eval_step]

    for step in steps:
        state = step(state)
        weave.publish(state, name=f"{step.__name__}-output")
        print(f"  {step.__name__:20s} -> {len(state.model_dump(exclude_none=True))} fields set")

    print()
    print(f"Final state: {state.metrics}")
    print(f"Full lineage in one object: {state.model_dump_json(indent=2, exclude_none=True)}")
    print(f"View: https://wandb.ai/valencelabs/{cfg.project}/weave")

    return state


# -- Hydra-zen CLI --

PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipeline")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipeline", config_path=None, version_base=None)
