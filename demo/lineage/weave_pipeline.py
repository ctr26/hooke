#!/usr/bin/env python
"""W&B Weave pipeline — native step tracking + caching.

Weave provides:
- @weave.op() — tracks inputs/outputs/code
- Automatic caching (same inputs → cached result)
- Lineage graph in W&B UI
- Code versioning built-in

pip install weave
"""

from pathlib import Path

import weave
from pydantic import BaseModel


# =============================================================================
# Schemas (still useful for type safety)
# =============================================================================


class ConditioningInput(BaseModel):
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]


class PretrainInput(BaseModel):
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    vocab_size: int
    conditioning_path: str


class FinetuneInput(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    step: int
    test_compounds: list[str]


class EvalInput(BaseModel):
    checkpoint_path: str
    target_cell_type: str
    test_compounds: list[str]


# =============================================================================
# Steps — @weave.op() instead of custom decorator
# =============================================================================


@weave.op()
def splits_step(output_dir: str) -> ConditioningInput:
    """Load/create data splits."""
    import json

    split_path = Path(output_dir) / "split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "train": ["cpd_001", "cpd_002"],
        "test": ["cpd_003"],
    }))

    return ConditioningInput(
        split_path=str(split_path),
        train_compounds=["cpd_001", "cpd_002"],
        test_compounds=["cpd_003"],
    )


@weave.op()
def conditioning_step(input: ConditioningInput, output_dir: str) -> PretrainInput:
    """Define model conditioning."""
    import json

    cond_path = Path(output_dir) / "conditioning.json"
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    cond_path.write_text(json.dumps({
        "cell_types": ["ARPE19", "HUVEC"],
        "vocab_size": 2048,
    }))

    return PretrainInput(
        split_path=input.split_path,
        train_compounds=input.train_compounds,
        test_compounds=input.test_compounds,
        cell_types=["ARPE19", "HUVEC"],
        vocab_size=2048,
        conditioning_path=str(cond_path),
    )


@weave.op()
def pretrain_step(input: PretrainInput, output_dir: str) -> FinetuneInput:
    """Pretrain model."""
    print(f"Training on {len(input.train_compounds)} compounds")

    ckpt_path = Path(output_dir) / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock checkpoint")

    return FinetuneInput(
        checkpoint_path=str(ckpt_path),
        cell_types=input.cell_types,
        step=200000,
        test_compounds=input.test_compounds,
    )


@weave.op()
def finetune_step(input: FinetuneInput, target_cell_type: str, output_dir: str) -> EvalInput:
    """Finetune model."""
    assert target_cell_type in input.cell_types

    ckpt_path = Path(output_dir) / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock finetuned")

    return EvalInput(
        checkpoint_path=str(ckpt_path),
        target_cell_type=target_cell_type,
        test_compounds=input.test_compounds,
    )


@weave.op()
def eval_step(input: EvalInput) -> dict:
    """Evaluate on test set."""
    print(f"Evaluating on {len(input.test_compounds)} test compounds")

    return {
        "map_cosine": 0.85,
        "pearson": 0.72,
    }


# =============================================================================
# Pipeline
# =============================================================================


def run_pipeline():
    """
    Weave tracks:
    - All inputs/outputs automatically
    - Code version at call time
    - Lineage graph (which ops produced which outputs)
    - Caches results for same inputs
    """
    # Initialize Weave project
    weave.init("hooke")

    output = "outputs/weave_demo"

    # Run pipeline — Weave tracks everything
    splits = splits_step(f"{output}/splits")
    config = conditioning_step(splits, f"{output}/cond")
    pretrain = pretrain_step(config, f"{output}/pretrain")
    finetune = finetune_step(pretrain, "ARPE19", f"{output}/ft")
    metrics = eval_step(finetune)

    print(f"Final: {metrics}")
    print("\nView lineage at: https://wandb.ai/<entity>/hooke/weave")


if __name__ == "__main__":
    run_pipeline()
