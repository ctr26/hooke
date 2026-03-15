#!/usr/bin/env python
"""Variant 1: Nested composition.

Each conf nests the previous conf. Full lineage is structural:
  result.inference.finetuning.pretrain.conditioning.data.split_file

Pros: Type-safe, each step has a unique input/output type.
Cons: Deep attribute access, schemas grow with pipeline length.

    uv run python demo/variants/nested.py
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel

from variants.schemas import (
    ConditioningConf,
    DataConf,
    EvalConf,
    FinetuningConf,
    InferenceConf,
    PretrainConf,
    ResultsConf,
)


@weave.op()
def splits_step(input: DataConf) -> ConditioningConf:
    return ConditioningConf(
        data=input,
        split_path=input.split_file,
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


@weave.op()
def conditioning_step(input: ConditioningConf) -> PretrainConf:
    return PretrainConf(
        conditioning=input,
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )


@weave.op()
def pretrain_step(input: PretrainConf) -> FinetuningConf:
    return FinetuningConf(
        pretrain=input,
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        step=200_000,
    )


@weave.op()
def finetuning_step(input: FinetuningConf) -> InferenceConf:
    return InferenceConf(
        finetuning=input,
        checkpoint_path=f"{input.checkpoint_path}.finetuned",
        step=input.step + 50_000,
    )


@weave.op()
def inference_step(input: InferenceConf) -> EvalConf:
    return EvalConf(
        inference=input,
        features_path="/outputs/features.npy",
        num_samples=len(input.finetuning.pretrain.conditioning.test_compounds) * 100,
    )


@weave.op()
def eval_step(input: EvalConf) -> ResultsConf:
    return ResultsConf(metrics={"map_cosine": 0.85, "pearson": 0.72})


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo-nested"


def run_pipeline(cfg: PipelineConfig) -> ResultsConf:
    weave.init(cfg.project)

    cond = splits_step(DataConf())
    pretrain = conditioning_step(cond)
    finetune = pretrain_step(pretrain)
    inference = finetuning_step(finetune)
    ev = inference_step(inference)
    result = eval_step(ev)

    weave.publish(result, name="results")
    print(f"Nested: {result.metrics}")
    print(f"Lineage: ev.inference.finetuning.pretrain.conditioning.data.split_file = {ev.inference.finetuning.pretrain.conditioning.data.split_file}")
    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="nested")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="nested", config_path=None, version_base=None)
