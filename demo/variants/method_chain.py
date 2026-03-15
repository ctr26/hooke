#!/usr/bin/env python
"""Variant 4: Method chaining (fluent API).

Steps are methods on the state object itself.
Pipeline reads like a sentence.

Pros: Discoverable via IDE autocomplete, self-documenting.
Cons: Mixes data and behaviour, harder to test steps in isolation.

    uv run python demo/variants/method_chain.py
"""

from __future__ import annotations

import weave
from hydra_zen import builds, store, zen
from pydantic import BaseModel


class Pipeline(BaseModel):
    # Data
    split_file: str = "data/splits/default.json"
    split_path: str | None = None
    train_compounds: list[str] | None = None
    val_compounds: list[str] | None = None
    test_compounds: list[str] | None = None

    # Conditioning
    cell_types: list[str] | None = None
    assay_types: list[str] | None = None
    vocab_size: int | None = None
    conditioning_path: str | None = None

    # Pretrain
    pretrain_checkpoint: str | None = None
    pretrain_step_count: int | None = None

    # Finetuning
    finetune_checkpoint: str | None = None
    finetune_step_count: int | None = None
    target_cell_type: str | None = None

    # Inference
    features_path: str | None = None
    num_samples: int | None = None

    # Eval
    metrics: dict[str, float] | None = None

    @weave.op()
    def split(self) -> Pipeline:
        return self.model_copy(update={
            "split_path": self.split_file,
            "train_compounds": ["cpd_001", "cpd_002", "cpd_003"],
            "val_compounds": ["cpd_004"],
            "test_compounds": ["cpd_005", "cpd_006"],
        })

    @weave.op()
    def condition(self) -> Pipeline:
        return self.model_copy(update={
            "cell_types": ["ARPE19", "HUVEC", "HepG2"],
            "assay_types": ["cell_paint", "brightfield"],
            "vocab_size": 2048,
            "conditioning_path": "/data/conditioning/v1.json",
        })

    @weave.op()
    def pretrain(self) -> Pipeline:
        return self.model_copy(update={
            "pretrain_checkpoint": "/checkpoints/pretrain_200k.pt",
            "pretrain_step_count": 200_000,
        })

    @weave.op()
    def finetune(self) -> Pipeline:
        return self.model_copy(update={
            "finetune_checkpoint": f"{self.pretrain_checkpoint}.finetuned",
            "finetune_step_count": (self.pretrain_step_count or 0) + 50_000,
            "target_cell_type": "ARPE19",
        })

    @weave.op()
    def infer(self) -> Pipeline:
        return self.model_copy(update={
            "features_path": "/outputs/features.npy",
            "num_samples": len(self.test_compounds or []) * 100,
        })

    @weave.op()
    def evaluate(self) -> Pipeline:
        return self.model_copy(update={
            "metrics": {"map_cosine": 0.85, "pearson": 0.72},
        })


class PipelineConfig(BaseModel):
    output_dir: str = "outputs/demo"
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
    print(f"Chain: {result.metrics}")
    return result


PipelineCfg = builds(PipelineConfig, populate_full_signature=True)
store(PipelineCfg, name="chain")
store.add_to_hydra_store()

if __name__ == "__main__":
    zen(run_pipeline).hydra_main(config_name="chain", config_path=None, version_base=None)
