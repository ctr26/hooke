from pathlib import Path

import polars as pl
import zarr
from pydantic import BaseModel, computed_field, field_validator


class PredictionsPaths(BaseModel):
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


class Predictions(BaseModel):
    """
    A predictions object.
    """

    paths: PredictionsPaths

    @field_validator("paths")
    def validate_paths(cls, v: PredictionsPaths) -> PredictionsPaths:
        if not v.obs_path.exists():
            raise ValueError(f"obs path {v.obs_path} does not exist")
        if not v.features_path.exists():
            raise ValueError(f"feature path {v.features_path} does not exist")
        return v

    @computed_field
    @property
    def obs(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.obs_path)

    @property
    def X(self) -> zarr.Array:
        return zarr.open(self.paths.features_path)
