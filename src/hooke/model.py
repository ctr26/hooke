"""Model wrappers for Hooke pipelines."""

from __future__ import annotations

import importlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from hooke.config import ModelConfig


class HookeBaseModel(ABC):
    """Abstract base class for all Hooke models."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._loaded = False

    @abstractmethod
    def load_weights(self, path: str | Path) -> None:
        """Load model weights from disk."""

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on inputs."""

    @abstractmethod
    def to(self, device: str) -> None:
        """Move model to a device (no-op for sklearn models)."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class HookeModel(HookeBaseModel):
    """Concrete model wrapping a scikit-learn estimator.

    Uses pickle for sklearn model serialization, which is the standard
    approach for scikit-learn. Only load weights from trusted sources.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._estimator: Any = None

    def load_weights(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        with open(path, "rb") as f:
            self._estimator = pickle.load(f)  # noqa: S301
        self._loaded = True

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self._loaded or self._estimator is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")
        import numpy as np

        X = np.asarray(inputs["features"])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = self._estimator.predict(X)
        return {"predictions": predictions.tolist()}

    def to(self, device: str) -> None:
        # No-op for sklearn models
        pass


def load_model_class(class_path: str) -> type[HookeBaseModel]:
    """Dynamically import a model class from a dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, HookeBaseModel)):
        raise TypeError(f"{class_path} is not a subclass of HookeBaseModel")
    return cls
