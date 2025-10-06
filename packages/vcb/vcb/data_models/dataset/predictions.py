from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class PredictionPaths(BaseModel):
    """
    The expected paths for a predictions directory.
    """

    root: Path

    @field_validator("root")
    def validate_root(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"root {v} is not a directory or does not exist")
        return v

    @computed_field
    @property
    def obs_path(self) -> Path:
        return self.root / "obs.parquet"

    @computed_field
    @property
    def features_path(self) -> Path:
        return self.root / "features.zarr"


class InMemoryPredictions(AnnotatedDataMatrix):
    """
    A baseline predictions object.

    Allows baseline predictions to be used for evaluation directly
    without having to save them to file.
    """

    obs: pl.DataFrame
    X: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def obs(self) -> pl.DataFrame:
        return self.obs

    @property
    def X(self) -> np.ndarray:
        return self.X
