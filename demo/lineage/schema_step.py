"""Schema-governed pipeline steps with W&B lineage.

Each step:
1. Has a Pydantic schema defining inputs/outputs
2. Produces W&B artifacts matching the output schema
3. Next step consumes artifacts matching input schema
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import wandb
from pydantic import BaseModel


# =============================================================================
# Schemas define step contracts
# =============================================================================


class ConditioningOutput(BaseModel):
    """Output schema for conditioning step."""

    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    config_path: str  # W&B artifact path


class PretrainOutput(BaseModel):
    """Output schema for pretraining step."""

    checkpoint_path: str  # W&B artifact path
    conditioning: ConditioningOutput  # Embedded from previous step
    step: int
    loss: float


class FinetuneOutput(BaseModel):
    """Output schema for finetuning step."""

    checkpoint_path: str
    base_checkpoint: PretrainOutput  # Embedded from previous step
    target_cell_type: str
    epochs: int


# =============================================================================
# Step decorator: schema → W&B artifact
# =============================================================================

T = TypeVar("T", bound=BaseModel)


def step(output_schema: type[T], artifact_type: str = "step_output"):
    """Decorator that wraps step output in W&B artifact.

    The step function must return an instance of output_schema.
    The artifact is logged with schema fields as metadata.
    """

    def decorator(fn):
        def wrapper(*args, **kwargs) -> T:
            # Run the step
            output: T = fn(*args, **kwargs)

            # Log output as W&B artifact
            artifact = wandb.Artifact(
                name=fn.__name__,
                type=artifact_type,
                metadata=output.model_dump(),
            )

            # Add any file paths as artifact files
            for field_name, value in output.model_dump().items():
                if field_name.endswith("_path") and isinstance(value, str):
                    path = Path(value)
                    if path.exists():
                        artifact.add_file(str(path))

            wandb.log_artifact(artifact)
            return output

        return wrapper

    return decorator


# =============================================================================
# Steps: schema-governed
# =============================================================================


@step(ConditioningOutput, artifact_type="config")
def conditioning_step(output_dir: Path) -> ConditioningOutput:
    """Produce conditioning config."""
    import json

    config = {
        "cell_types": ["ARPE19", "HUVEC", "HepG2"],
        "assay_types": ["cell_paint", "brightfield"],
        "vocab_size": 2048,
    }

    config_path = output_dir / "conditioning.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    return ConditioningOutput(
        cell_types=config["cell_types"],
        assay_types=config["assay_types"],
        vocab_size=config["vocab_size"],
        config_path=str(config_path),
    )


@step(PretrainOutput, artifact_type="model")
def pretrain_step(conditioning: ConditioningOutput, output_dir: Path) -> PretrainOutput:
    """Pretrain using conditioning output."""
    print(f"Pretraining with cell types: {conditioning.cell_types}")

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("mock checkpoint")

    return PretrainOutput(
        checkpoint_path=str(checkpoint_path),
        conditioning=conditioning,
        step=200000,
        loss=0.1,
    )


@step(FinetuneOutput, artifact_type="model")
def finetune_step(
    pretrain: PretrainOutput, target_cell_type: str, output_dir: Path
) -> FinetuneOutput:
    """Finetune using pretrain output."""
    print(f"Finetuning from step {pretrain.step} for {target_cell_type}")

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("mock finetuned")

    return FinetuneOutput(
        checkpoint_path=str(checkpoint_path),
        base_checkpoint=pretrain,
        target_cell_type=target_cell_type,
        epochs=20,
    )


# =============================================================================
# Pipeline: chain steps via schema outputs
# =============================================================================


def run_pipeline():
    """Run pipeline: each step consumes previous step's output."""
    output = Path("outputs/schema_demo")

    with wandb.init(project="hooke", job_type="conditioning"):
        cond_output = conditioning_step(output / "conditioning")

    with wandb.init(project="hooke", job_type="pretrain"):
        # Consume conditioning output directly
        pretrain_output = pretrain_step(cond_output, output / "pretrain")

    with wandb.init(project="hooke", job_type="finetune"):
        # Consume pretrain output directly
        finetune_output = finetune_step(pretrain_output, "ARPE19", output / "finetune")

    print("\nPipeline complete!")
    print(f"Final output: {finetune_output.model_dump_json(indent=2)}")


if __name__ == "__main__":
    run_pipeline()
