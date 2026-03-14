from pathlib import Path

from pydantic import (
    BaseModel,
    computed_field,
    field_validator,
)


class DatasetDirectory(BaseModel):
    """
    Utility class to automatically parse any dataset directory that is formatted according to the official schema.
    """

    root: Path

    @field_validator("root")
    def validate_root(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"root {v} is not a directory or does not exist")
        return v

    @computed_field
    @property
    def dataset_id(self) -> str:
        return self.root.name

    @computed_field
    @property
    def obs_path(self) -> Path:
        return self.root / f"{self.dataset_id}_obs.parquet"

    @computed_field
    @property
    def features_path(self) -> Path:
        return self.root / f"{self.dataset_id}_features.zarr"

    @computed_field
    @property
    def metadata_path(self) -> Path:
        return self.root / f"{self.dataset_id}_dataset_metadata.json"

    @computed_field
    @property
    def var_path(self) -> Path:
        return self.root / f"{self.dataset_id}_var.parquet"
