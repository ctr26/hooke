#!/usr/bin/env python
"""Variant 3: Functional pipe (reduce).

Same accumulator state but composed with functools.reduce.
Pipeline is a data structure, not control flow.

Pros: Pipeline definition is declarative, easy to serialize/inspect.
Cons: Same loss of per-step types as accumulator.

    uv run python demo/variants/functional_pipe.py
"""

from functools import reduce
from typing import Callable

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel

from variants.schemas import PipelineState

Step = Callable[[PipelineState], PipelineState]


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


def pipe(steps: list[Step], initial: PipelineState) -> PipelineState:
    """Apply steps sequentially via reduce."""
    return reduce(lambda state, step: step(state), steps, initial)


PIPELINE: list[Step] = [
    splits_step,
    conditioning_step,
    pretrain_step,
    finetuning_step,
    inference_step,
    eval_step,
]


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo-pipe"


def run_pipeline(cfg: PipelineConfig) -> PipelineState:
    weave.init(cfg.project)

    result = pipe(PIPELINE, PipelineState())

    weave.publish(result, name="results")
    print(f"Pipe: {result.metrics}")
    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipe")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipe", config_path=None, version_base=None)
