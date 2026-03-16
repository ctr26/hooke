"""End-to-end pipeline for model loading and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hooke.config import PipelineConfig
from hooke.model import HookeBaseModel, load_model_class
from hooke.preprocessing import PreprocessingPipeline, Preprocessor, StandardScaler

PREPROCESSOR_REGISTRY: dict[str, type[Preprocessor]] = {
    "standard_scaler": StandardScaler,
}


def _build_preprocessor(steps_config: list[dict[str, Any]]) -> PreprocessingPipeline:
    """Build a preprocessing pipeline from config dicts."""
    steps: list[Preprocessor] = []
    for step_cfg in steps_config:
        name = step_cfg.get("name", "")
        if name not in PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Unknown preprocessor: '{name}'. Available: {list(PREPROCESSOR_REGISTRY)}"
            )
        steps.append(PREPROCESSOR_REGISTRY[name]())
    return PreprocessingPipeline(steps)


class Pipeline:
    """End-to-end inference pipeline: config → preprocess → model → output."""

    def __init__(
        self,
        config: PipelineConfig,
        model: HookeBaseModel,
        preprocessor: PreprocessingPipeline,
    ) -> None:
        self.config = config
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_config(cls, path: str | Path) -> Pipeline:
        """Load a pipeline from a YAML config file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        config = PipelineConfig(**raw)

        # Load and instantiate model
        model_cls = load_model_class(config.model.model_class)
        model = model_cls(config.model)

        if config.model.weights_path is not None:
            model.load_weights(config.model.weights_path)

        model.to(config.model.device)

        # Build preprocessing
        preprocessor = _build_preprocessor(config.data.preprocessing_steps)

        return cls(config=config, model=model, preprocessor=preprocessor)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the full pipeline: preprocess → predict."""
        if "features" in inputs:
            features = np.asarray(inputs["features"])
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if self.preprocessor.steps:
                features = self.preprocessor.transform(features)
            inputs = {**inputs, "features": features}
        return self.model.predict(inputs)
