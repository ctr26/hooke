"""Pipeline — demo stub."""

from typing import Any

from hooke_forge.config import PipelineConfig
from hooke_forge.model_loader import TorchModelWrapper


class Pipeline:
    def __init__(self, model: TorchModelWrapper, config: PipelineConfig) -> None:
        self.model = model
        self.config = config

    @classmethod
    def from_config(cls, path: str) -> "Pipeline":
        pass

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.model.predict(inputs)
