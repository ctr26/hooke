"""Shared fixtures for Hooke tests.

Uses pickle for sklearn model serialization (standard practice for tests).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge


@pytest.fixture
def toy_weights(tmp_path: Path) -> Path:
    """Create a toy sklearn model and save weights."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    weights_path = tmp_path / "model.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(model, f)
    return weights_path


@pytest.fixture
def example_config_path(tmp_path: Path, toy_weights: Path) -> Path:
    """Create an example YAML config file pointing to toy weights."""
    config_path = tmp_path / "config.yaml"
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
  preprocessing_steps: []

name: test-pipeline
version: "0.1.0"
""")
    return config_path
