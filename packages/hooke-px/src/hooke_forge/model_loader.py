"""Typed model loading — demo stub."""

from abc import ABC, abstractmethod
from typing import Any

from hooke_forge.config import ModelConfig


class BaseModelWrapper(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def load_weights(self, path: str) -> None: ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def to(self, device: str) -> "BaseModelWrapper": ...


class TorchModelWrapper(BaseModelWrapper):
    def load_weights(self, path: str) -> None:
        pass

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}

    def to(self, device: str) -> "TorchModelWrapper":
        return self
