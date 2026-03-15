#!/usr/bin/env python
"""Variant 4: Method chaining (fluent API).

Steps are methods on the state object. Pipeline reads like a sentence.

    uv run python demo/variants/method_chain.py
"""

from __future__ import annotations

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


class Pipeline(BaseModel):
    # Data
    split_file: str = "data/splits/default.json"
    train_compounds: list[str] | None = None
    val_compounds: list[str] | None = None
    test_compounds: list[str] | None = None

    # Conditioning
    cell_types: list[str] | None = None
    assay_types: list[str] | None = None
    vocab_size: int | None = None
    conditioning_weights: list[float] | None = None

    # Pretrain
    model_weights: list[float] | None = None
    pretrain_step_count: int | None = None

    # Finetuning
    finetuned_weights: list[float] | None = None
    finetune_step_count: int | None = None
    target_cell_type: str | None = None

    # Inference
    features: list[list[float]] | None = None
    num_samples: int | None = None

    # Eval
    metrics: dict[str, float] | None = None

    @weave.op()
    def split(self) -> Pipeline:
        return self.model_copy(update={
            "train_compounds": ["cpd_001", "cpd_002", "cpd_003"],
            "val_compounds": ["cpd_004"],
            "test_compounds": ["cpd_005", "cpd_006"],
        })

    @weave.op()
    def condition(self) -> Pipeline:
        weights = [0.1 * i for i in range(len(self.train_compounds or []))]
        return self.model_copy(update={
            "cell_types": ["ARPE19", "HUVEC", "HepG2"],
            "assay_types": ["cell_paint", "brightfield"],
            "vocab_size": 2048,
            "conditioning_weights": weights,
        })

    @weave.op()
    def pretrain(self) -> Pipeline:
        weights = [w + 1.0 for w in (self.conditioning_weights or [])] + [0.5] * (self.vocab_size or 0)
        return self.model_copy(update={
            "model_weights": weights[:10],
            "pretrain_step_count": 200_000,
        })

    @weave.op()
    def finetune(self) -> Pipeline:
        weights = [w * 1.1 for w in (self.model_weights or [])]
        return self.model_copy(update={
            "finetuned_weights": weights,
            "finetune_step_count": (self.pretrain_step_count or 0) + 50_000,
            "target_cell_type": "ARPE19",
        })

    @weave.op()
    def infer(self) -> Pipeline:
        n = len(self.test_compounds or [])
        weights = self.finetuned_weights or self.model_weights or []
        features = [[w * (i + 1) for w in weights[:3]] for i in range(n)]
        return self.model_copy(update={
            "features": features,
            "num_samples": n,
        })

    @weave.op()
    def evaluate(self) -> Pipeline:
        features = self.features or []
        mean_feat = sum(f[0] for f in features) / len(features) if features else 0
        return self.model_copy(update={
            "metrics": {"map_cosine": round(mean_feat, 4), "pearson": 0.72},
        })


class PipelineConfig(BaseModel):
    project: str = "hooke-demo-chain"


def run_pipeline(cfg: PipelineConfig) -> Pipeline:
    weave.init(cfg.project)

    result = (
        Pipeline()
        .split()
        .condition()
        .pretrain()
        .finetune()
        .infer()
        .evaluate()
    )

    weave.publish(result, name="results")

    print("=== Method Chain ===")
    print(f"  Metrics: {result.metrics}")
    print(f"  Features shape: {len(result.features or [])}x{len((result.features or [[]])[0])}")
    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="chain")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="chain", config_path=None, version_base=None)
