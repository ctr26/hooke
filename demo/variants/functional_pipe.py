#!/usr/bin/env python
"""Variant 3: Functional pipe (reduce).

Pipeline is a data structure composed with functools.reduce.

    uv run python demo/variants/functional_pipe.py
"""

from functools import reduce
from typing import Callable

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel

from variants.schemas import PipelineState
from variants.accumulator import (
    splits_step,
    conditioning_step,
    pretrain_step,
    finetuning_step,
    inference_step,
    eval_step,
)

Step = Callable[[PipelineState], PipelineState]


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
    project: str = "hooke-demo-pipe"


def run_pipeline(cfg: PipelineConfig) -> PipelineState:
    weave.init(cfg.project)

    result = pipe(PIPELINE, PipelineState())
    weave.publish(result, name="results")

    print("=== Functional Pipe (reduce) ===")
    print(f"  Steps: {' -> '.join(s.__name__ for s in PIPELINE)}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Features shape: {len(result.features or [])}x{len((result.features or [[]])[0])}")

    # Partial pipeline demo
    partial = pipe(PIPELINE[:3], PipelineState())
    print(f"\n  Partial (first 3 steps): model_weights={partial.model_weights is not None}, metrics={partial.metrics}")

    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipe")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipe", config_path=None, version_base=None)
