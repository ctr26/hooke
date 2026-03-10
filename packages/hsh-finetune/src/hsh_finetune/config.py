"""Finetuning configuration.

Kept in the hsh-finetune workspace member so it can be installed
and versioned independently of the core hsh package.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hsh.config import TrainConfig


class FinetuneConfig(BaseModel, frozen=True):
    """Finetuning config -- composes TrainConfig with a base checkpoint.

    Finetuning reuses the pretraining loop; this config just adds
    the checkpoint path and overrides training defaults.
    """

    base_checkpoint: str = Field(description="Path to pretrained checkpoint")
    train: TrainConfig = Field(
        default_factory=lambda: TrainConfig(num_steps=50, lr=1e-4, output_dir="./finetune_outputs")
    )
