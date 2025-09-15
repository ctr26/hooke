from pathlib import Path

from pydantic import BaseModel, computed_field, field_validator


class PredictionPaths(BaseModel):
    """
    The expected paths for a predictions directory.
    """

    root: Path
    var_path: Path

    @field_validator("root")
    def validate_root(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"root {v} is not a directory or does not exist")
        return v

    @field_validator("var_path")
    def validate_var_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"var_path {v} does not exist")
        return v

    @computed_field
    @property
    def obs_path(self) -> Path:
        return self.root / "obs.parquet"

    @computed_field
    @property
    def features_path(self) -> Path:
        return self.root / "features.zarr"
