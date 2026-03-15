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


class SplitsInput(BaseModel):
    split_file: str = "data/splits/default.json"
    output_dir: str = "outputs/splits"


class ConditioningInput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainInput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class InferenceInput(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int
    test_compounds: list[str]


class EvalInput(BaseModel):
    features_path: str
    num_samples: int
    split_path: str


class EvalOutput(BaseModel):
    metrics: dict[str, float]


# -- Steps: @weave.op() for lineage tracking --


@weave.op()
def splits_step(input: SplitsInput) -> ConditioningInput:
    return ConditioningInput(
        split_path=input.split_file,
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


@weave.op()
def conditioning_step(input: ConditioningInput) -> PretrainInput:
    return PretrainInput(
        **input.model_dump(),
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )


@weave.op()
def pretrain_step(input: PretrainInput) -> InferenceInput:
    return InferenceInput(
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200_000,
        test_compounds=input.test_compounds,
    )


@weave.op()
def inference_step(input: InferenceInput) -> EvalInput:
    return EvalInput(
        features_path="/outputs/features.npy",
        num_samples=len(input.test_compounds) * 100,
        split_path="/data/splits/v1.json",
    )


@weave.op()
def eval_step(input: EvalInput) -> EvalOutput:
    return EvalOutput(metrics={"map_cosine": 0.85, "pearson": 0.72})


# -- Pipeline --


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo"


def run_pipeline(cfg: PipelineConfig) -> EvalOutput:
    weave.init(cfg.project)

    cond_in = splits_step(SplitsInput())
    pretrain_in = conditioning_step(cond_in)
    inference_in = pretrain_step(pretrain_in)
    eval_in = inference_step(inference_in)
    result = eval_step(eval_in)

    print("Pipeline:")
    print(f"  splits_step(SplitsInput)           -> {type(cond_in).__name__}")
    print(f"  conditioning_step(ConditioningInput) -> {type(pretrain_in).__name__}")
    print(f"  pretrain_step(PretrainInput)        -> {type(inference_in).__name__}")
    print(f"  inference_step(InferenceInput)      -> {type(eval_in).__name__}")
    print(f"  eval_step(EvalInput)                -> {result.metrics}")
    print()
    print(f"Chain proof: pretrain_in.split_path == cond_in.split_path -> {pretrain_in.split_path == cond_in.split_path}")
    print(f"JSON roundtrip: {InferenceInput.model_validate_json(inference_in.model_dump_json()) == inference_in}")
    print(f"Weave project: {cfg.project}")

    return result


# -- Hydra-zen CLI --

PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipeline")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipeline", config_path=None, version_base=None)
