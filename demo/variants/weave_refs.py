#!/usr/bin/env python
"""Variant 5: Weave refs — artifact-mediated pipeline.

Steps publish outputs as named Weave objects. Downstream steps
consume via weave.ref() instead of direct Python variable passing.
This is how you'd chain steps across separate runs or projects.

Pros: Steps are fully decoupled, can run independently, cross-project.
Cons: Requires Weave backend, not purely local.

    uv run python demo/variants/weave_refs.py
"""

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


def _publish(obj, name: str):
    """Publish only when Weave client is active."""
    try:
        weave.publish(obj, name=name)
    except Exception:
        pass


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
    result = ConditioningConf(
        data=input,
        split_path=input.split_file,
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )
    _publish(result, name="splits-output")
    return result


@weave.op()
def conditioning_step(input: ConditioningConf) -> PretrainConf:
    result = PretrainConf(
        conditioning=input,
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )
    _publish(result, name="conditioning-output")
    return result


@weave.op()
def pretrain_step(input: PretrainConf) -> FinetuningConf:
    result = FinetuningConf(
        pretrain=input,
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        step=200_000,
    )
    _publish(result, name="pretrain-output")
    return result


@weave.op()
def finetuning_step(input: FinetuningConf) -> InferenceConf:
    result = InferenceConf(
        finetuning=input,
        checkpoint_path=f"{input.checkpoint_path}.finetuned",
        step=input.step + 50_000,
    )
    _publish(result, name="finetuning-output")
    return result


@weave.op()
def inference_step(input: InferenceConf) -> EvalConf:
    result = EvalConf(
        inference=input,
        features_path="/outputs/features.npy",
        num_samples=len(input.finetuning.pretrain.conditioning.test_compounds) * 100,
    )
    _publish(result, name="inference-output")
    return result


@weave.op()
def eval_step(input: EvalConf) -> ResultsConf:
    result = ResultsConf(metrics={"map_cosine": 0.85, "pearson": 0.72})
    _publish(result, name="results")
    return result


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
    project: str = "hooke-demo-refs"


def run_pipeline(cfg: PipelineConfig) -> ResultsConf:
    """Run full pipeline — each step publishes, could be separate runs."""
    weave.init(cfg.project)

    cond = splits_step(DataConf())
    pretrain = conditioning_step(cond)
    finetune = pretrain_step(pretrain)
    inference = finetuning_step(finetune)
    ev = inference_step(inference)
    result = eval_step(ev)

    print(f"Refs: {result.metrics}")
    return result


def run_from_ref(cfg: PipelineConfig, from_step: str = "pretrain") -> ResultsConf:
    """Resume pipeline from a published artifact — the real power of refs.

    Example: pretrain is done, just re-run finetuning + inference + eval.
    """
    weave.init(cfg.project)

    ref = weave.ref(f"{from_step}-output:latest").get()
    print(f"Loaded {from_step}-output from Weave: {type(ref).__name__}")

    if from_step == "pretrain":
        finetune = finetuning_step(ref)
        inference = inference_step(finetune)
        ev = eval_step(inference)
    elif from_step == "finetuning":
        inference = inference_step(ref)
        ev = eval_step(inference)
    elif from_step == "inference":
        ev = eval_step(ref)
    else:
        raise ValueError(f"Unknown step: {from_step}")

    print(f"Resumed from {from_step}: {ev.metrics}")
    return ev


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="refs")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="refs", config_path=None, version_base=None)
