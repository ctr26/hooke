#!/usr/bin/env python
"""Variant 5: Weave refs — artifact-mediated pipeline.

Steps publish outputs as named Weave objects. Downstream steps
can consume via weave.ref() for cross-run/cross-project chaining.

    uv run python demo/variants/weave_refs.py
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


def _publish(obj, name: str):
    """Publish only when Weave client is active."""
    try:
        weave.publish(obj, name=name)
    except Exception:
        pass


@weave.op()
def splits_step(input: DataConf) -> ConditioningConf:
    result = ConditioningConf(
        data=input,
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )
    _publish(result, name="splits-output")
    return result


@weave.op()
def conditioning_step(input: ConditioningConf) -> PretrainConf:
    weights = [0.1 * i for i in range(len(input.train_compounds))]
    result = PretrainConf(
        conditioning=input,
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_weights=weights,
    )
    _publish(result, name="conditioning-output")
    return result


@weave.op()
def pretrain_step(input: PretrainConf) -> FinetuningConf:
    weights = [w + 1.0 for w in input.conditioning_weights] + [0.5] * input.vocab_size
    result = FinetuningConf(
        pretrain=input,
        model_weights=weights[:10],
        step=200_000,
    )
    _publish(result, name="pretrain-output")
    return result


@weave.op()
def finetuning_step(input: FinetuningConf) -> InferenceConf:
    weights = [w * 1.1 for w in input.model_weights]
    result = InferenceConf(
        finetuning=input,
        model_weights=weights,
        step=input.step + 50_000,
    )
    _publish(result, name="finetuning-output")
    return result


@weave.op()
def inference_step(input: InferenceConf) -> EvalConf:
    n = len(input.finetuning.pretrain.conditioning.test_compounds)
    features = [[w * (i + 1) for w in input.model_weights[:3]] for i in range(n)]
    result = EvalConf(
        inference=input,
        features=features,
        num_samples=n,
    )
    _publish(result, name="inference-output")
    return result


@weave.op()
def eval_step(input: EvalConf) -> ResultsConf:
    mean_feat = sum(f[0] for f in input.features) / len(input.features)
    result = ResultsConf(metrics={"map_cosine": round(mean_feat, 4), "pearson": 0.72})
    _publish(result, name="results")
    return result


class PipelineConfig(BaseModel):
    project: str = "hooke-demo-refs"


def run_pipeline(cfg: PipelineConfig) -> ResultsConf:
    """Run full pipeline — each step publishes its output."""
    weave.init(cfg.project)

    cond = splits_step(DataConf())
    pretrain = conditioning_step(cond)
    finetune = pretrain_step(pretrain)
    inference = finetuning_step(finetune)
    ev = inference_step(inference)
    result = eval_step(ev)

    print("=== Weave Refs ===")
    print(f"  Published: splits-output, conditioning-output, pretrain-output, finetuning-output, inference-output, results")
    print(f"  Metrics: {result.metrics}")
    print(f"  Resume from any step: weave.ref('pretrain-output:latest').get()")
    return result


def run_from_ref(cfg: PipelineConfig, from_step: str = "pretrain") -> ResultsConf:
    """Resume pipeline from a published artifact."""
    weave.init(cfg.project)

    ref = weave.ref(f"{from_step}-output:latest").get()
    print(f"  Loaded {from_step}-output: {type(ref).__name__}")

    if from_step == "pretrain":
        inference = finetuning_step(ref)
        ev = inference_step(inference)
        return eval_step(ev)
    elif from_step == "finetuning":
        ev = inference_step(ref)
        return eval_step(ev)
    elif from_step == "inference":
        return eval_step(ref)
    else:
        raise ValueError(f"Unknown step: {from_step}")


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="refs")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="refs", config_path=None, version_base=None)
