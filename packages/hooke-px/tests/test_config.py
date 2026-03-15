"""Tests for config schemas."""

import pytest
from pydantic import ValidationError

from hooke_forge.config import DataConfig, ModelConfig, PipelineConfig


class TestModelConfig:
    def test_valid(self):
        m = ModelConfig(
            model_class="hooke_forge.model.architecture.HookeForge",
            weights_path="/ckpt.pt",
        )
        assert m.device == "cuda"
        assert m.precision == "float32"

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_class="some.Model")


class TestDataConfig:
    def test_defaults(self):
        d = DataConfig(dataset_path="/data")
        assert d.batch_size == 32
        assert d.preprocessing_steps == []

    def test_custom(self):
        d = DataConfig(
            dataset_path="/data",
            batch_size=128,
            preprocessing_steps=["normalize", "augment"],
        )
        assert len(d.preprocessing_steps) == 2


class TestPipelineConfig:
    def test_nested(self):
        p = PipelineConfig(
            model=ModelConfig(
                model_class="hooke_forge.model.architecture.HookeForge",
                weights_path="/ckpt.pt",
            ),
            data=DataConfig(dataset_path="/data"),
        )
        assert p.project == "hooke-px"
        assert p.model.device == "cuda"

    def test_from_dict(self):
        raw = {
            "model": {
                "model_class": "some.Model",
                "weights_path": "/ckpt.pt",
            },
            "data": {"dataset_path": "/data"},
        }
        p = PipelineConfig(**raw)
        assert p.output_dir == "outputs"

    def test_serialization_roundtrip(self):
        p = PipelineConfig(
            model=ModelConfig(model_class="m", weights_path="/w"),
            data=DataConfig(dataset_path="/d"),
        )
        rebuilt = PipelineConfig.model_validate_json(p.model_dump_json())
        assert rebuilt == p
