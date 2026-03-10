"""Pydantic configs for all pipeline stages.

Config definitions live in Python as pydantic-validated types.
No YAML config files -- "YAML out, not in."
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel, frozen=True):
    """Architecture config for the minimal flow-matching model."""

    input_dim: int = Field(default=32, gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)


class TrainConfig(BaseModel, frozen=True):
    """Training hyperparameters."""

    num_steps: int = Field(default=100, gt=0)
    batch_size: int = Field(default=8, gt=0)
    lr: float = Field(default=1e-3, gt=0)
    num_samples: int = Field(default=64, gt=0, description="Synthetic dataset size")
    ckpt_every: int = Field(default=50, gt=0)
    eval_every: int = Field(default=50, gt=0)
    output_dir: str = Field(default="./outputs")
    seed: int = Field(default=42)


class FinetuneConfig(BaseModel, frozen=True):
    """Finetuning config -- composes TrainConfig with a base checkpoint.

    Finetuning reuses the pretraining loop; this config just adds
    the checkpoint path and overrides training defaults.
    """

    base_checkpoint: str = Field(description="Path to pretrained checkpoint")
    train: TrainConfig = Field(
        default_factory=lambda: TrainConfig(num_steps=50, lr=1e-4, output_dir="./finetune_outputs")
    )


class EvalConfig(BaseModel, frozen=True):
    """Evaluation config."""

    checkpoint: str = Field(description="Path to checkpoint file")
    batch_size: int = Field(default=8, gt=0)
    num_samples: int = Field(default=64, gt=0)
    output_dir: str = Field(default="./eval_outputs")


class InferConfig(BaseModel, frozen=True):
    """Inference config."""

    checkpoint: str = Field(description="Path to checkpoint file")
    batch_size: int = Field(default=8, gt=0)
    num_samples: int = Field(default=32, gt=0)
    output_dir: str = Field(default="./predictions")
