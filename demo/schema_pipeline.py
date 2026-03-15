#!/usr/bin/env python
"""Schema-governed pipeline demo.

Each step returns the next step's input as a Pydantic schema.
Weave tracks lineage, hydra-zen provides CLI config.

    uv run python demo/schema_pipeline.py
    uv run python demo/schema_pipeline.py --help
    uv run python demo/schema_pipeline.py output_dir=/tmp/my_run
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


# -- Schemas: each describes the input to a step --


class DataConf(BaseModel):
    split_file: str = "data/splits/default.json"
    output_dir: str = "outputs/splits"


class ConditioningConf(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainConf(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class FinetuningConf(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int
    test_compounds: list[str]
    target_cell_type: str = "ARPE19"


class InferenceConf(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int
    test_compounds: list[str]
    target_cell_type: str


class EvalConf(BaseModel):
    features_path: str
    num_samples: int
    split_path: str


class ResultsConf(BaseModel):
    metrics: dict[str, float]


# -- Steps: @weave.op() for lineage tracking --


@weave.op()
def splits_step(input: DataConf) -> ConditioningConf:
    return ConditioningConf(
        split_path=input.split_file,
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


@weave.op()
def conditioning_step(input: ConditioningConf) -> PretrainConf:
    return PretrainConf(
        **input.model_dump(),
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )


@weave.op()
def pretrain_step(input: PretrainConf) -> FinetuningConf:
    return FinetuningConf(
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200_000,
        test_compounds=input.test_compounds,
    )


@weave.op()
def finetuning_step(input: FinetuningConf) -> InferenceConf:
    return InferenceConf(
        checkpoint_path=f"{input.checkpoint_path}.finetuned",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=input.step + 50_000,
        test_compounds=input.test_compounds,
        target_cell_type=input.target_cell_type,
    )


@weave.op()
def inference_step(input: InferenceConf) -> EvalConf:
    return EvalConf(
        features_path="/outputs/features.npy",
        num_samples=len(input.test_compounds) * 100,
        split_path="/data/splits/v1.json",
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

    cond_in = splits_step(DataConf())
    pretrain_in = conditioning_step(cond_in)
    finetune_in = pretrain_step(pretrain_in)
    inference_in = finetuning_step(finetune_in)
    eval_in = inference_step(inference_in)
    result = eval_step(eval_in)

    print("Pipeline:")
    print(f"  splits_step(DataConf)              -> {type(cond_in).__name__}")
    print(f"  conditioning_step(ConditioningConf) -> {type(pretrain_in).__name__}")
    print(f"  pretrain_step(PretrainConf)         -> {type(finetune_in).__name__}")
    print(f"  finetuning_step(FinetuningConf)     -> {type(inference_in).__name__}")
    print(f"  inference_step(InferenceConf)       -> {type(eval_in).__name__}")
    print(f"  eval_step(EvalConf)                 -> {result.metrics}")
    print()
    print(f"Chain proof: pretrain_in.split_path == cond_in.split_path -> {pretrain_in.split_path == cond_in.split_path}")
    print(f"JSON roundtrip: {InferenceConf.model_validate_json(inference_in.model_dump_json()) == inference_in}")
    print(f"Weave project: {cfg.project}")

    return result


# -- Hydra-zen CLI --

PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipeline")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipeline", config_path=None, version_base=None)
