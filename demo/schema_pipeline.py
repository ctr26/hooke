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


# -- Schemas: step N returns step N+1's input --


class SplitsOutput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class ConditioningOutput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class PretrainOutput(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int
    test_compounds: list[str]


class InferenceOutput(BaseModel):
    features_path: str
    num_samples: int


class EvalOutput(BaseModel):
    metrics: dict[str, float]


# -- Steps: @weave.op() for lineage tracking --


@weave.op()
def splits_step() -> SplitsOutput:
    return SplitsOutput(
        split_path="/data/splits/v1.json",
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


@weave.op()
def conditioning_step(input: SplitsOutput) -> ConditioningOutput:
    return ConditioningOutput(
        **input.model_dump(),
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )


@weave.op()
def pretrain_step(input: ConditioningOutput) -> PretrainOutput:
    return PretrainOutput(
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200_000,
        test_compounds=input.test_compounds,
    )


@weave.op()
def inference_step(input: PretrainOutput) -> InferenceOutput:
    return InferenceOutput(
        features_path="/outputs/features.npy",
        num_samples=len(input.test_compounds) * 100,
    )


@weave.op()
def eval_step(features_path: str, split_path: str) -> EvalOutput:
    return EvalOutput(metrics={"map_cosine": 0.85, "pearson": 0.72})


# -- Pipeline --


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo"


def run_pipeline(cfg: PipelineConfig) -> EvalOutput:
    weave.init(cfg.project)

    splits = splits_step()
    cond = conditioning_step(splits)
    pretrain = pretrain_step(cond)
    inference = inference_step(pretrain)
    result = eval_step(inference.features_path, splits.split_path)

    print("Pipeline:")
    print(f"  splits    -> {type(splits).__name__}")
    print(f"  condition -> {type(cond).__name__}")
    print(f"  pretrain  -> {type(pretrain).__name__}")
    print(f"  inference -> {type(inference).__name__}")
    print(f"  eval      -> {result.metrics}")
    print()
    print(f"Chain proof: cond.split_path == splits.split_path -> {cond.split_path == splits.split_path}")
    print(f"JSON roundtrip: {PretrainOutput.model_validate_json(pretrain.model_dump_json()) == pretrain}")
    print(f"Weave project: {cfg.project}")

    return result


# -- Hydra-zen CLI --

PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="pipeline")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="pipeline", config_path=None, version_base=None)
