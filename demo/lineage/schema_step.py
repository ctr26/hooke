"""Schema-governed pipeline: define input requirements, not outputs.

Each step defines what it NEEDS (input schema with non-optional fields).
Previous step must produce data satisfying those requirements.
"""

from pathlib import Path
from typing import Any

import wandb
from pydantic import BaseModel


# =============================================================================
# Input schemas: what each step REQUIRES
# =============================================================================


class PretrainInput(BaseModel):
    """What pretrain_step needs."""

    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    config_path: str


class FinetuneInput(BaseModel):
    """What finetune_step needs."""

    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int


class EvalInput(BaseModel):
    """What eval_step needs."""

    checkpoint_path: str
    target_cell_type: str


# =============================================================================
# Step decorator
# =============================================================================


def step(artifact_type: str = "step_output"):
    """Log step output to W&B."""

    def decorator(fn):
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            output = fn(*args, **kwargs)

            artifact = wandb.Artifact(
                name=fn.__name__,
                type=artifact_type,
                metadata=output,
            )

            for key, value in output.items():
                if key.endswith("_path") and isinstance(value, str):
                    path = Path(value)
                    if path.exists():
                        artifact.add_file(str(path))

            wandb.log_artifact(artifact)
            return output

        return wrapper

    return decorator


# =============================================================================
# Steps: validate input, return dict
# =============================================================================


@step(artifact_type="config")
def conditioning_step(output_dir: Path) -> dict:
    """Produces data for PretrainInput."""
    import json

    config_path = output_dir / "conditioning.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"cell_types": ["ARPE19", "HUVEC"]}))

    return {
        "cell_types": ["ARPE19", "HUVEC"],
        "assay_types": ["cell_paint"],
        "vocab_size": 2048,
        "config_path": str(config_path),
    }


@step(artifact_type="model")
def pretrain_step(input: PretrainInput, output_dir: Path) -> dict:
    """Requires: PretrainInput. Produces data for FinetuneInput."""
    print(f"Building with vocab_size={input.vocab_size}")

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock")

    return {
        "checkpoint_path": str(ckpt_path),
        "cell_types": input.cell_types,
        "vocab_size": input.vocab_size,
        "step": 200000,
        "loss": 0.1,
    }


@step(artifact_type="model")
def finetune_step(input: FinetuneInput, target_cell_type: str, output_dir: Path) -> dict:
    """Requires: FinetuneInput. Produces data for EvalInput."""
    assert target_cell_type in input.cell_types

    ckpt_path = output_dir / "checkpoint.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("mock finetuned")

    return {
        "checkpoint_path": str(ckpt_path),
        "target_cell_type": target_cell_type,
        "base_step": input.step,
        "epochs": 20,
    }


@step(artifact_type="metrics")
def eval_step(input: EvalInput) -> dict:
    """Requires: EvalInput. Final output."""
    return {
        "checkpoint_path": input.checkpoint_path,
        "metrics": {"map_cosine": 0.85, "pearson": 0.72},
    }


# =============================================================================
# Pipeline: validate at boundaries
# =============================================================================


def run_pipeline():
    """
    conditioning_step() → dict
                           ↓ validate as PretrainInput
    pretrain_step(PretrainInput) → dict
                                    ↓ validate as FinetuneInput
    finetune_step(FinetuneInput) → dict
                                    ↓ validate as EvalInput
    eval_step(EvalInput) → dict
    """
    output = Path("outputs/schema_demo")

    with wandb.init(project="hooke", job_type="conditioning"):
        cond_data = conditioning_step(output / "cond")

    with wandb.init(project="hooke", job_type="pretrain"):
        # Validate: does cond_data satisfy PretrainInput?
        pretrain_in = PretrainInput(**cond_data)
        pretrain_data = pretrain_step(pretrain_in, output / "pretrain")

    with wandb.init(project="hooke", job_type="finetune"):
        # Validate: does pretrain_data satisfy FinetuneInput?
        finetune_in = FinetuneInput(**pretrain_data)
        finetune_data = finetune_step(finetune_in, "ARPE19", output / "ft")

    with wandb.init(project="hooke", job_type="eval"):
        # Validate: does finetune_data satisfy EvalInput?
        eval_in = EvalInput(**finetune_data)
        result = eval_step(eval_in)

    print(f"Final: {result['metrics']}")


if __name__ == "__main__":
    run_pipeline()
