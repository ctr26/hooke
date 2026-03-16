"""Tests for config validation (happy + error paths)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hooke.config import DataConfig, ModelConfig, PipelineConfig


class TestModelConfig:
    def test_valid_config(self) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel", device="cpu")
        assert cfg.model_class == "hooke.model.HookeModel"
        assert cfg.device == "cpu"
        assert cfg.precision == "float32"
        assert cfg.weights_path is None

    def test_invalid_precision(self) -> None:
        with pytest.raises(ValidationError, match="Precision must be one of"):
            ModelConfig(model_class="hooke.model.HookeModel", precision="int8")

    def test_nonexistent_weights_path(self, tmp_path) -> None:
        with pytest.raises(ValidationError, match="Weights file not found"):
            ModelConfig(
                model_class="hooke.model.HookeModel",
                weights_path=tmp_path / "nonexistent.pkl",
            )

    def test_frozen(self) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel")
        with pytest.raises(ValidationError):
            cfg.device = "cuda"


class TestDataConfig:
    def test_valid_config(self) -> None:
        cfg = DataConfig(feature_names=["a", "b"])
        assert cfg.batch_size == 32
        assert cfg.preprocessing_steps == []

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValidationError, match="batch_size must be >= 1"):
            DataConfig(feature_names=["a"], batch_size=0)

    def test_negative_batch_size(self) -> None:
        with pytest.raises(ValidationError, match="batch_size must be >= 1"):
            DataConfig(feature_names=["a"], batch_size=-5)


class TestPipelineConfig:
    def test_valid_config(self) -> None:
        cfg = PipelineConfig(
            model=ModelConfig(model_class="hooke.model.HookeModel"),
            data=DataConfig(feature_names=["x"]),
        )
        assert cfg.name == "default"
        assert cfg.version == "0.1.0"

    def test_empty_feature_names(self) -> None:
        with pytest.raises(ValidationError, match="feature_names must not be empty"):
            PipelineConfig(
                model=ModelConfig(model_class="hooke.model.HookeModel"),
                data=DataConfig(feature_names=[]),
            )
