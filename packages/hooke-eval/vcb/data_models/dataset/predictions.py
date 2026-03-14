from pathlib import Path

from pydantic import BaseModel, computed_field, field_validator


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
        path = self.root / "obs.parquet"
        if not path.exists():
            raise ValueError(f"obs.parquet not found at {path}")
        return path

    @computed_field
    @property
    def features_path(self) -> Path:
        path = self.root / "features.zarr"
        if not path.exists():
            raise ValueError(f"features.zarr not found at {path}")
        return path
