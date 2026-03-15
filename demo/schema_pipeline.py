#!/usr/bin/env python
"""Schema-governed pipeline demo.

Each step returns the next step's input as a Pydantic schema.
Schemas compose by nesting the previous conf, not flattening.
Weave tracks lineage, hydra-zen provides CLI config.

    uv run python demo/schema_pipeline.py
    uv run python demo/schema_pipeline.py --help
    uv run python demo/schema_pipeline.py output_dir=/tmp/my_run
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


# -- Schemas: nested composition --


class DataConf(BaseModel):
    split_file: str = "data/splits/default.json"
    output_dir: str = "outputs/splits"


class ConditioningConf(BaseModel):
    data: DataConf
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainConf(BaseModel):
    conditioning: ConditioningConf
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class FinetuningConf(BaseModel):
    pretrain: PretrainConf
    checkpoint_path: str
    step: int
    target_cell_type: str = "ARPE19"


class InferenceConf(BaseModel):
    finetuning: FinetuningConf
    checkpoint_path: str
    step: int


class EvalConf(BaseModel):
    inference: InferenceConf
    features_path: str
    num_samples: int


class ResultsConf(BaseModel):
    metrics: dict[str, float]


# -- Steps: @weave.op() for lineage tracking --


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


# -- Pipeline --


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo"


def run_pipeline(cfg: PipelineConfig) -> ResultsConf:
    weave.init(cfg.project)

    cond = splits_step(DataConf())
    weave.publish(cond, name="splits-output")

    pretrain = conditioning_step(cond)
    weave.publish(pretrain, name="conditioning-output")

    finetune = pretrain_step(pretrain)
    weave.publish(finetune, name="pretrain-output")

    inference = finetuning_step(finetune)
    weave.publish(inference, name="finetuning-output")

    ev = inference_step(inference)
    weave.publish(ev, name="inference-output")

    result = eval_step(ev)
    weave.publish(result, name="results")

    print("Pipeline:")
    print(f"  splits_step(DataConf)              -> {type(cond).__name__}")
    print(f"  conditioning_step(ConditioningConf) -> {type(pretrain).__name__}")
    print(f"  pretrain_step(PretrainConf)         -> {type(finetune).__name__}")
    print(f"  finetuning_step(FinetuningConf)     -> {type(inference).__name__}")
    print(f"  inference_step(InferenceConf)       -> {type(ev).__name__}")
    print(f"  eval_step(EvalConf)                 -> {result.metrics}")
    print()

    # Traverse the nested lineage
    print(f"Lineage traversal: ev.inference.finetuning.pretrain.conditioning.data.split_file")
    print(f"  = {ev.inference.finetuning.pretrain.conditioning.data.split_file}")
    print()
    print(f"Weave project: {cfg.project}")
    print(f"View lineage: https://wandb.ai/valencelabs/{cfg.project}/weave")

    return result


# -- Hydra-zen CLI --

PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipeline")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipeline", config_path=None, version_base=None)
