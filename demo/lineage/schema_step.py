"""Schema-governed pipeline: each step returns next step's input schema.

conditioning_step() → PretrainInput
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


class PretrainInput(BaseModel):
    """conditioning_step returns this."""

    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    config_path: str


class FinetuneInput(BaseModel):
    """pretrain_step returns this."""

    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int


class EvalInput(BaseModel):
    """finetune_step returns this."""

    checkpoint_path: str
    target_cell_type: str


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
# Steps: return next step's input schema
# =============================================================================


@step(artifact_type="config")
def conditioning_step(output_dir: Path) -> PretrainInput:
    """Returns PretrainInput."""
    import json

    config_path = output_dir / "conditioning.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"cell_types": ["ARPE19", "HUVEC"]}))

    return PretrainInput(
        cell_types=["ARPE19", "HUVEC"],
        assay_types=["cell_paint"],
        vocab_size=2048,
        config_path=str(config_path),
    )


@step(artifact_type="model")
def pretrain_step(input: PretrainInput, output_dir: Path) -> FinetuneInput:
    """Takes PretrainInput, returns FinetuneInput."""
    print(f"Building with vocab_size={input.vocab_size}")

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock")

    return FinetuneInput(
        checkpoint_path=str(ckpt_path),
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200000,
    )


@step(artifact_type="model")
def finetune_step(input: FinetuneInput, target_cell_type: str, output_dir: Path) -> EvalInput:
    """Takes FinetuneInput, returns EvalInput."""
    assert target_cell_type in input.cell_types

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock finetuned")

    return EvalInput(
        checkpoint_path=str(ckpt_path),
        target_cell_type=target_cell_type,
    )


@step(artifact_type="metrics")
def eval_step(input: EvalInput) -> Result:
    """Takes EvalInput, returns Result."""
    return Result(
        metrics={"map_cosine": 0.85, "pearson": 0.72},
    )


# =============================================================================
# Pipeline: direct chaining, no validation needed
# =============================================================================


def run_pipeline():
    """
    conditioning_step() → PretrainInput
                              ↓
    pretrain_step(PretrainInput) → FinetuneInput
                                       ↓
    finetune_step(FinetuneInput) → EvalInput
                                       ↓
    eval_step(EvalInput) → Result
    """
    output = Path("outputs/schema_demo")

    with wandb.init(project="hooke", job_type="conditioning"):
        pretrain_in: PretrainInput = conditioning_step(output / "cond")

    with wandb.init(project="hooke", job_type="pretrain"):
        finetune_in: FinetuneInput = pretrain_step(pretrain_in, output / "pretrain")

    with wandb.init(project="hooke", job_type="finetune"):
        eval_in: EvalInput = finetune_step(finetune_in, "ARPE19", output / "ft")

    with wandb.init(project="hooke", job_type="eval"):
        result: Result = eval_step(eval_in)

    print(f"Final: {result.metrics}")


if __name__ == "__main__":
    run_pipeline()
