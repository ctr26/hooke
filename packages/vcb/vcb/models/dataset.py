from pathlib import Path

import polars as pl
import zarr
from pydantic import BaseModel, computed_field, field_validator, model_validator


class DatasetPaths(BaseModel):
    """
    The expected paths for a dataset directory.
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


class DatasetMetadata(BaseModel):
    """
    A dataset metadata object.

    Note (cwognum): There is additional metadata that is not included here, since it's not used in this code base.
    """

    dataset_id: str
    biological_context: list[str]


class Dataset(BaseModel):
    """
    A dataset.
    """

    paths: DatasetPaths

    @field_validator("paths")
    def validate_paths(cls, v: DatasetPaths) -> DatasetPaths:
        if not v.obs_path.exists():
            raise ValueError(f"obs path {v.obs_path} does not exist")
        if not v.features_path.exists():
            raise ValueError(f"feature path {v.features_path} does not exist")
        if not v.metadata_path.exists():
            raise ValueError(f"metadata path {v.metadata_path} does not exist")
        if not v.var_path.exists():
            raise ValueError(f"var path {v.var_path} does not exist")
        return v

    @model_validator(mode="after")
    def validate_dataset_id(self) -> "Dataset":
        if self.dataset_id != self.paths.dataset_id:
            raise ValueError(
                f"dataset_id {self.dataset_id} does not match paths.dataset_id {self.paths.dataset_id}"
            )
        return self

    @property
    def dataset_id(self) -> str:
        return self.metadata.dataset_id

    @property
    def obs(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.obs_path)

    @property
    def var(self) -> zarr.Array:
        return zarr.open(self.paths.var_path)

    @property
    def X(self) -> zarr.Array:
        return zarr.open(self.paths.features_path)

    @computed_field
    @property
    def metadata(self) -> DatasetMetadata:
        with open(self.paths.metadata_path, "r") as fd:
            metadata = DatasetMetadata.model_validate_json(fd.read())
        return metadata
