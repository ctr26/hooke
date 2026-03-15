#!/usr/bin/env python
"""Variant 1: Nested composition.

Each conf nests the previous conf. Full lineage is structural.
Outputs are actual data (weights, features), not file paths.

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
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


@weave.op()
def conditioning_step(input: ConditioningConf) -> PretrainConf:
    # Simulate: produce conditioning weights from compound data
    weights = [0.1 * i for i in range(len(input.train_compounds))]
    return PretrainConf(
        conditioning=input,
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_weights=weights,
    )


@weave.op()
def pretrain_step(input: PretrainConf) -> FinetuningConf:
    # Simulate: produce model weights from pretraining
    weights = [w + 1.0 for w in input.conditioning_weights] + [0.5] * input.vocab_size
    return FinetuningConf(
        pretrain=input,
        model_weights=weights[:10],  # truncate for demo
        step=200_000,
    )


@weave.op()
def finetuning_step(input: FinetuningConf) -> InferenceConf:
    # Simulate: refine weights for target cell type
    weights = [w * 1.1 for w in input.model_weights]
    return InferenceConf(
        finetuning=input,
        model_weights=weights,
        step=input.step + 50_000,
    )


@weave.op()
def inference_step(input: InferenceConf) -> EvalConf:
    # Simulate: produce feature embeddings per test compound
    n_compounds = len(input.finetuning.pretrain.conditioning.test_compounds)
    features = [[w * (i + 1) for w in input.model_weights[:3]] for i in range(n_compounds)]
    return EvalConf(
        inference=input,
        features=features,
        num_samples=n_compounds,
    )


@weave.op()
def eval_step(input: EvalConf) -> ResultsConf:
    # Simulate: compute metrics from features
    mean_feat = sum(f[0] for f in input.features) / len(input.features)
    return ResultsConf(metrics={"map_cosine": round(mean_feat, 4), "pearson": 0.72})


class PipelineConfig(BaseModel):
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

    print("=== Nested Composition ===")
    print(f"  Conditioning weights: {pretrain.conditioning_weights}")
    print(f"  Pretrain weights (first 5): {finetune.model_weights[:5]}")
    print(f"  Finetuned weights (first 5): {inference.model_weights[:5]}")
    print(f"  Features shape: {len(ev.features)}x{len(ev.features[0])}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Lineage: ev.inference.finetuning.pretrain.conditioning.data.split_file = {ev.inference.finetuning.pretrain.conditioning.data.split_file}")
    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="nested")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="nested", config_path=None, version_base=None)
