"""Tests for end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hooke.pipeline import Pipeline


class TestPipeline:
    def test_from_config(self, example_config_path: Path) -> None:
        pipe = Pipeline.from_config(example_config_path)
        assert pipe.model.is_loaded
        assert pipe.config.name == "test-pipeline"

    def test_call(self, example_config_path: Path) -> None:
        pipe = Pipeline.from_config(example_config_path)
        result = pipe({"features": [1.0, 2.0, 3.0]})
        assert "predictions" in result
        assert isinstance(result["predictions"], list)

    def test_call_batch(self, example_config_path: Path) -> None:
        pipe = Pipeline.from_config(example_config_path)
        X = np.random.default_rng(0).standard_normal((10, 3))
        result = pipe({"features": X})
        assert len(result["predictions"]) == 10

    def test_nonexistent_config(self) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Pipeline.from_config("/nonexistent/config.yaml")

    def test_with_preprocessing(self, tmp_path: Path, toy_weights: Path) -> None:
        config_path = tmp_path / "config_preproc.yaml"
        config_path.write_text(f"""\
model:
  model_class: hooke.model.HookeModel
  weights_path: "{toy_weights}"
  device: cpu
  precision: float32

data:
  feature_names:
    - feat_1
    - feat_2
    - feat_3
  batch_size: 16
  preprocessing_steps:
    - name: standard_scaler

name: preproc-pipeline
""")
        pipe = Pipeline.from_config(config_path)
        assert len(pipe.preprocessor.steps) == 1

        # Fit the preprocessor before calling
        X_train = np.random.default_rng(42).standard_normal((20, 3))
        pipe.preprocessor.fit(X_train)

        result = pipe({"features": [1.0, 2.0, 3.0]})
        assert "predictions" in result
