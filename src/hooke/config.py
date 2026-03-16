"""Configuration schemas for Hooke pipelines (Pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ModelConfig(BaseModel):
    """Configuration for model instantiation and loading."""

    model_config = ConfigDict(frozen=True)

    model_class: str
    weights_path: Path | None = None
    device: str = "cpu"
    precision: str = "float32"

    @field_validator("weights_path")
    @classmethod
    def validate_weights_path(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Weights file not found: {v}")
        return v

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        allowed = {"float16", "float32", "float64", "bfloat16"}
        if v not in allowed:
            raise ValueError(f"Precision must be one of {allowed}, got '{v}'")
        return v


class DataConfig(BaseModel):
    """Configuration for data input and preprocessing."""

    model_config = ConfigDict(frozen=True)

    feature_names: list[str]
    batch_size: int = 32
    preprocessing_steps: list[dict[str, Any]] = []

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v


class PipelineConfig(BaseModel):
    """Combined configuration for an end-to-end pipeline."""

    model_config = ConfigDict(frozen=True)

    model: ModelConfig
    data: DataConfig
    name: str = "default"
    version: str = "0.1.0"

    @model_validator(mode="after")
    def validate_feature_names_not_empty(self) -> PipelineConfig:
        if not self.data.feature_names:
            raise ValueError("data.feature_names must not be empty")
        return self
