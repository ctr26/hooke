"""Schema-governed pipeline: output of step N = input of step N+1.

Each step:
1. Takes previous step's Output schema as input
2. Returns its own Output schema
3. @step decorator logs to W&B
"""

from pathlib import Path
from typing import TypeVar

import wandb
from pydantic import BaseModel


# =============================================================================
# Schemas: Output of step N = Input of step N+1
# =============================================================================


class ConditioningOutput(BaseModel):
    """conditioning_step output → pretrain_step input."""

    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    config_path: str


class PretrainOutput(BaseModel):
    """pretrain_step output → finetune_step input."""

    checkpoint_path: str
    cell_types: list[str]  # Carried forward for validation
    vocab_size: int
    step: int
    loss: float


class FinetuneOutput(BaseModel):
    """finetune_step output → eval_step input."""

    checkpoint_path: str
    base_step: int
    target_cell_type: str
    epochs: int


class EvalOutput(BaseModel):
    """eval_step output → final result."""

    checkpoint_path: str
    metrics: dict[str, float]


# =============================================================================
# Step decorator
# =============================================================================

T = TypeVar("T", bound=BaseModel)


def step(output_schema: type[T], artifact_type: str = "step_output"):
    """Wrap step: validate output schema, log to W&B."""

    def decorator(fn):
        def wrapper(*args, **kwargs) -> T:
            output: T = fn(*args, **kwargs)

            # Log as W&B artifact with schema as metadata
            artifact = wandb.Artifact(
                name=fn.__name__,
                type=artifact_type,
                metadata=output.model_dump(),
            )

            for field, value in output.model_dump().items():
                if field.endswith("_path") and isinstance(value, str):
                    path = Path(value)
                    if path.exists():
                        artifact.add_file(str(path))

            wandb.log_artifact(artifact)
            return output

        return wrapper

    return decorator


# =============================================================================
# Steps: input schema = previous output schema
# =============================================================================


@step(ConditioningOutput, artifact_type="config")
def conditioning_step(output_dir: Path) -> ConditioningOutput:
    """Produce conditioning config."""
    import json

    config_path = output_dir / "conditioning.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"cell_types": ["ARPE19", "HUVEC"], "vocab_size": 2048})
    )

    return ConditioningOutput(
        cell_types=["ARPE19", "HUVEC"],
        assay_types=["cell_paint"],
        vocab_size=2048,
        config_path=str(config_path),
    )


@step(PretrainOutput, artifact_type="model")
def pretrain_step(input: ConditioningOutput, output_dir: Path) -> PretrainOutput:
    """Input: ConditioningOutput. Output: PretrainOutput."""
    print(f"Building model with vocab_size={input.vocab_size}")

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock")

    return PretrainOutput(
        checkpoint_path=str(ckpt_path),
        cell_types=input.cell_types,  # Carry forward
        vocab_size=input.vocab_size,
        step=200000,
        loss=0.1,
    )


@step(FinetuneOutput, artifact_type="model")
def finetune_step(
    input: PretrainOutput, target_cell_type: str, output_dir: Path
) -> FinetuneOutput:
    """Input: PretrainOutput. Output: FinetuneOutput."""
    assert target_cell_type in input.cell_types, f"{target_cell_type} not in vocab"

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock finetuned")

    return FinetuneOutput(
        checkpoint_path=str(ckpt_path),
        base_step=input.step,
        target_cell_type=target_cell_type,
        epochs=20,
    )


@step(EvalOutput, artifact_type="metrics")
def eval_step(input: FinetuneOutput) -> EvalOutput:
    """Input: FinetuneOutput. Output: EvalOutput."""
    return EvalOutput(
        checkpoint_path=input.checkpoint_path,
        metrics={"map_cosine": 0.85, "pearson": 0.72},
    )


# =============================================================================
# Pipeline: chain via matching schemas
# =============================================================================


def run_pipeline():
    """
    conditioning_step() → ConditioningOutput
                              ↓
    pretrain_step(ConditioningOutput) → PretrainOutput
                                            ↓
    finetune_step(PretrainOutput) → FinetuneOutput
                                        ↓
    eval_step(FinetuneOutput) → EvalOutput
    """
    output = Path("outputs/schema_demo")

    with wandb.init(project="hooke", job_type="conditioning"):
        cond: ConditioningOutput = conditioning_step(output / "cond")

    with wandb.init(project="hooke", job_type="pretrain"):
        pretrain: PretrainOutput = pretrain_step(cond, output / "pretrain")

    with wandb.init(project="hooke", job_type="finetune"):
        finetune: FinetuneOutput = finetune_step(pretrain, "ARPE19", output / "ft")

    with wandb.init(project="hooke", job_type="eval"):
        result: EvalOutput = eval_step(finetune)

    print(f"Final: {result.metrics}")


if __name__ == "__main__":
    run_pipeline()
