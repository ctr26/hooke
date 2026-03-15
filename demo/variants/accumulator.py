#!/usr/bin/env python
"""Variant 2: Accumulator with for loop.

Single PipelineState grows at each step. Outputs are actual data.

    uv run python demo/variants/accumulator.py
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel

from variants.schemas import PipelineState


@weave.op()
def splits_step(state: PipelineState) -> PipelineState:
    return state.model_copy(update={
        "train_compounds": ["cpd_001", "cpd_002", "cpd_003"],
        "val_compounds": ["cpd_004"],
        "test_compounds": ["cpd_005", "cpd_006"],
    })


@weave.op()
def conditioning_step(state: PipelineState) -> PipelineState:
    weights = [0.1 * i for i in range(len(state.train_compounds or []))]
    return state.model_copy(update={
        "cell_types": ["ARPE19", "HUVEC", "HepG2"],
        "assay_types": ["cell_paint", "brightfield"],
        "vocab_size": 2048,
        "conditioning_weights": weights,
    })


@weave.op()
def pretrain_step(state: PipelineState) -> PipelineState:
    weights = [w + 1.0 for w in (state.conditioning_weights or [])] + [0.5] * (state.vocab_size or 0)
    return state.model_copy(update={
        "model_weights": weights[:10],
        "pretrain_step": 200_000,
    })


@weave.op()
def finetuning_step(state: PipelineState) -> PipelineState:
    weights = [w * 1.1 for w in (state.model_weights or [])]
    return state.model_copy(update={
        "finetuned_weights": weights,
        "finetune_step": (state.pretrain_step or 0) + 50_000,
        "target_cell_type": "ARPE19",
    })


@weave.op()
def inference_step(state: PipelineState) -> PipelineState:
    n = len(state.test_compounds or [])
    weights = state.finetuned_weights or state.model_weights or []
    features = [[w * (i + 1) for w in weights[:3]] for i in range(n)]
    return state.model_copy(update={
        "features": features,
        "num_samples": n,
    })


@weave.op()
def eval_step(state: PipelineState) -> PipelineState:
    features = state.features or []
    mean_feat = sum(f[0] for f in features) / len(features) if features else 0
    return state.model_copy(update={
        "metrics": {"map_cosine": round(mean_feat, 4), "pearson": 0.72},
    })


STEPS = [splits_step, conditioning_step, pretrain_step, finetuning_step, inference_step, eval_step]


class PipelineConfig(BaseModel):
    project: str = "hooke-demo-accumulator"


def run_pipeline(cfg: PipelineConfig) -> PipelineState:
    weave.init(cfg.project)

    state = PipelineState()
    print("=== Accumulator (for loop) ===")
    for step in STEPS:
        state = step(state)
        weave.publish(state, name=f"{step.__name__}-output")
        print(f"  {step.__name__:20s} -> {len(state.model_dump(exclude_none=True))} fields set")

    print(f"\n  Metrics: {state.metrics}")
    print(f"  Features shape: {len(state.features or [])}x{len((state.features or [[]])[0])}")
    return state


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="accumulator")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="accumulator", config_path=None, version_base=None)
