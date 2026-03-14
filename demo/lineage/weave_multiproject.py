#!/usr/bin/env python
"""Multi-project Weave pipeline with Pydantic schemas.

Each workspace = separate Weave project = isolated code versioning.
Cross-project refs maintain lineage.

packages/
├── hooke-px/    → weave project "hooke-px"
├── hooke-tx/    → weave project "hooke-tx"
└── hooke-eval/  → weave project "hooke-eval"
"""

import weave
from pydantic import BaseModel


# =============================================================================
# Shared schemas (in a shared package or duplicated)
# =============================================================================


class SplitsOutput(BaseModel):
    """Output of splits step → input for conditioning."""
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]


class ConditioningOutput(BaseModel):
    """Output of conditioning → input for pretrain."""
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    vocab_size: int


class PretrainOutput(BaseModel):
    """Output of pretrain (hooke-px) → input for eval (hooke-eval)."""
    checkpoint_path: str
    cell_types: list[str]
    step: int
    test_compounds: list[str]


class EvalOutput(BaseModel):
    """Output of eval."""
    metrics: dict[str, float]


# =============================================================================
# hooke-px workspace
# =============================================================================


def run_hooke_px():
    """hooke-px project: splits → conditioning → pretrain."""
    weave.init("hooke-px")

    @weave.op()
    def splits_step() -> SplitsOutput:
        return SplitsOutput(
            split_path="/data/splits/v1.json",
            train_compounds=["cpd_001", "cpd_002"],
            test_compounds=["cpd_003"],
        )

    @weave.op()
    def conditioning_step(input: SplitsOutput) -> ConditioningOutput:
        return ConditioningOutput(
            split_path=input.split_path,
            train_compounds=input.train_compounds,
            test_compounds=input.test_compounds,
            cell_types=["ARPE19", "HUVEC"],
            vocab_size=2048,
        )

    @weave.op()
    def pretrain_step(input: ConditioningOutput) -> PretrainOutput:
        print(f"Pretrain on {len(input.train_compounds)} compounds")
        return PretrainOutput(
            checkpoint_path="/checkpoints/pretrain_v1.pt",
            cell_types=input.cell_types,
            step=200000,
            test_compounds=input.test_compounds,
        )

    # Run pipeline
    splits = splits_step()
    config = conditioning_step(splits)
    checkpoint = pretrain_step(config)

    # Publish for cross-project use
    weave.publish(checkpoint, name="pretrain-output")
    print(f"Published: hooke-px/pretrain-output")

    return checkpoint


# =============================================================================
# hooke-eval workspace
# =============================================================================


def run_hooke_eval(checkpoint_ref: str = "hooke-px/pretrain-output:latest"):
    """hooke-eval project: eval checkpoint from hooke-px."""
    weave.init("hooke-eval")

    @weave.op()
    def eval_step(input: PretrainOutput) -> EvalOutput:
        print(f"Eval on {len(input.test_compounds)} test compounds")
        print(f"Checkpoint: {input.checkpoint_path}")
        return EvalOutput(
            metrics={"map_cosine": 0.85, "pearson": 0.72}
        )

    # Get checkpoint from hooke-px project
    checkpoint: PretrainOutput = weave.ref(checkpoint_ref).get()

    # Eval — lineage crosses project boundary
    result = eval_step(checkpoint)

    weave.publish(result, name="eval-output")
    print(f"Published: hooke-eval/eval-output")

    return result


# =============================================================================
# Orchestrator (runs both)
# =============================================================================


def run_full_pipeline():
    """
    hooke-px/splits_step → SplitsOutput
                               ↓
    hooke-px/conditioning_step → ConditioningOutput
                                     ↓
    hooke-px/pretrain_step → PretrainOutput
                                  ↓
                    weave.publish("pretrain-output")
                                  ↓
                    weave.ref("hooke-px/pretrain-output")
                                  ↓
    hooke-eval/eval_step → EvalOutput
    """
    # Run hooke-px
    checkpoint = run_hooke_px()

    # Run hooke-eval (consumes from hooke-px)
    result = run_hooke_eval()

    print(f"\nFinal: {result.metrics}")
    print("\nLineage spans both projects:")
    print("  hooke-px: https://wandb.ai/<entity>/hooke-px/weave")
    print("  hooke-eval: https://wandb.ai/<entity>/hooke-eval/weave")


if __name__ == "__main__":
    run_full_pipeline()
