"""Schema-governed pipeline: each step returns next step's input schema.

Conditioning and data splits are tracked as artifacts in the lineage.

splits_step() → ConditioningInput
conditioning_step(ConditioningInput) → PretrainInput
pretrain_step(PretrainInput) → FinetuneInput
finetune_step(FinetuneInput) → EvalInput
eval_step(EvalInput) → Result
"""

from pathlib import Path
from typing import TypeVar

import wandb
from pydantic import BaseModel


# =============================================================================
# Schemas: step N returns step N+1's input
# =============================================================================


class ConditioningInput(BaseModel):
    """splits_step returns this."""

    split_path: str  # Path to split JSON
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainInput(BaseModel):
    """conditioning_step returns this."""

    # From splits
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]

    # From conditioning
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class FinetuneInput(BaseModel):
    """pretrain_step returns this."""

    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int

    # Carry forward for eval
    test_compounds: list[str]


class EvalInput(BaseModel):
    """finetune_step returns this."""

    checkpoint_path: str
    target_cell_type: str
    test_compounds: list[str]


class Result(BaseModel):
    """eval_step returns this."""

    metrics: dict[str, float]


# =============================================================================
# Step decorator
# =============================================================================

T = TypeVar("T", bound=BaseModel)


def step(artifact_type: str = "step_output"):
    """Log step output to W&B."""

    def decorator(fn):
        def wrapper(*args, **kwargs) -> T:
            output: T = fn(*args, **kwargs)

            artifact = wandb.Artifact(
                name=fn.__name__,
                type=artifact_type,
                metadata=output.model_dump(),
            )

            for key, value in output.model_dump().items():
                if key.endswith("_path") and isinstance(value, str):
                    path = Path(value)
                    if path.exists():
                        artifact.add_file(str(path))

            wandb.log_artifact(artifact)
            return output

        return wrapper

    return decorator


# =============================================================================
# Steps
# =============================================================================


@step(artifact_type="split")
def splits_step(split_file: Path, output_dir: Path) -> ConditioningInput:
    """Load/create data splits. Returns ConditioningInput."""
    import json

    # In practice: load from split_file or create new split
    split_data = {
        "train": ["compound_001", "compound_002", "compound_003"],
        "val": ["compound_004"],
        "test": ["compound_005", "compound_006"],
    }

    split_path = output_dir / "split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_data))

    return ConditioningInput(
        split_path=str(split_path),
        train_compounds=split_data["train"],
        val_compounds=split_data["val"],
        test_compounds=split_data["test"],
    )


@step(artifact_type="config")
def conditioning_step(input: ConditioningInput, output_dir: Path) -> PretrainInput:
    """Define model conditioning. Returns PretrainInput."""
    import json

    conditioning = {
        "cell_types": ["ARPE19", "HUVEC", "HepG2"],
        "assay_types": ["cell_paint", "brightfield"],
        "vocab_size": 2048,
    }

    cond_path = output_dir / "conditioning.json"
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    cond_path.write_text(json.dumps(conditioning))

    return PretrainInput(
        # Carry forward splits
        split_path=input.split_path,
        train_compounds=input.train_compounds,
        val_compounds=input.val_compounds,
        test_compounds=input.test_compounds,
        # Add conditioning
        cell_types=conditioning["cell_types"],
        assay_types=conditioning["assay_types"],
        vocab_size=conditioning["vocab_size"],
        conditioning_path=str(cond_path),
    )


@step(artifact_type="model")
def pretrain_step(input: PretrainInput, output_dir: Path) -> FinetuneInput:
    """Pretrain model. Returns FinetuneInput."""
    print(f"Training on {len(input.train_compounds)} compounds")
    print(f"Vocab size: {input.vocab_size}")

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock")

    return FinetuneInput(
        checkpoint_path=str(ckpt_path),
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200000,
        test_compounds=input.test_compounds,  # Carry forward for eval
    )


@step(artifact_type="model")
def finetune_step(input: FinetuneInput, target_cell_type: str, output_dir: Path) -> EvalInput:
    """Finetune model. Returns EvalInput."""
    assert target_cell_type in input.cell_types

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock finetuned")

    return EvalInput(
        checkpoint_path=str(ckpt_path),
        target_cell_type=target_cell_type,
        test_compounds=input.test_compounds,
    )


@step(artifact_type="metrics")
def eval_step(input: EvalInput) -> Result:
    """Evaluate on test set. Returns Result."""
    print(f"Evaluating on {len(input.test_compounds)} test compounds")

    return Result(
        metrics={"map_cosine": 0.85, "pearson": 0.72},
    )


# =============================================================================
# Pipeline
# =============================================================================


def run_pipeline():
    """
    splits_step() → ConditioningInput
                        ↓
    conditioning_step(ConditioningInput) → PretrainInput
                                               ↓
    pretrain_step(PretrainInput) → FinetuneInput
                                       ↓
    finetune_step(FinetuneInput) → EvalInput
                                       ↓
    eval_step(EvalInput) → Result
    """
    output = Path("outputs/schema_demo")
    split_file = Path("data/splits/default.json")

    with wandb.init(project="hooke", job_type="splits"):
        cond_in: ConditioningInput = splits_step(split_file, output / "splits")

    with wandb.init(project="hooke", job_type="conditioning"):
        pretrain_in: PretrainInput = conditioning_step(cond_in, output / "cond")

    with wandb.init(project="hooke", job_type="pretrain"):
        finetune_in: FinetuneInput = pretrain_step(pretrain_in, output / "pretrain")

    with wandb.init(project="hooke", job_type="finetune"):
        eval_in: EvalInput = finetune_step(finetune_in, "ARPE19", output / "ft")

    with wandb.init(project="hooke", job_type="eval"):
        result: Result = eval_step(eval_in)

    print(f"Final: {result.metrics}")


if __name__ == "__main__":
    run_pipeline()
