"""Tests for model loading and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hooke.config import ModelConfig
from hooke.model import HookeModel, load_model_class


class TestHookeModel:
    def test_predict_without_loading(self) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel")
        model = HookeModel(cfg)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict({"features": [1.0, 2.0, 3.0]})

    def test_load_and_predict(self, toy_weights: Path) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel", weights_path=toy_weights)
        model = HookeModel(cfg)
        model.load_weights(toy_weights)
        assert model.is_loaded

        result = model.predict({"features": [1.0, 2.0, 3.0]})
        assert "predictions" in result
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 1

    def test_load_nonexistent_weights(self) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel")
        model = HookeModel(cfg)
        with pytest.raises(FileNotFoundError):
            model.load_weights("/nonexistent/path.pkl")

    def test_predict_batch(self, toy_weights: Path) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel", weights_path=toy_weights)
        model = HookeModel(cfg)
        model.load_weights(toy_weights)
        X = np.random.default_rng(0).standard_normal((5, 3))
        result = model.predict({"features": X})
        assert len(result["predictions"]) == 5

    def test_to_device_noop(self, toy_weights: Path) -> None:
        cfg = ModelConfig(model_class="hooke.model.HookeModel", weights_path=toy_weights)
        model = HookeModel(cfg)
        model.to("cuda")  # should not raise


class TestLoadModelClass:
    def test_load_hooke_model(self) -> None:
        cls = load_model_class("hooke.model.HookeModel")
        assert cls is HookeModel

    def test_load_invalid_path(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_model_class("nonexistent.module.Model")

    def test_load_non_model_class(self) -> None:
        with pytest.raises(TypeError, match="not a subclass"):
            load_model_class("hooke.config.ModelConfig")
