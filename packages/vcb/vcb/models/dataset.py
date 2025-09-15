from pathlib import Path

from pydantic import (
    BaseModel,
    computed_field,
    field_validator,
)

from vcb.models.anndata import AnnotatedDataMatrix


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


class DatasetMetadata(BaseModel):
    """
    A dataset metadata object.

    Note (cwognum): There is additional metadata that is not included here, since it's not used in this code base.
    """

    dataset_id: str
    biological_context: list[str]


class Dataset(AnnotatedDataMatrix):
    """
    Dataset.
    """

    metadata_path: Path

    @property
    def dataset_id(self) -> str:
        return self.metadata.dataset_id

    @computed_field
    @property
    def metadata(self) -> DatasetMetadata:
        with open(self.metadata_path, "r") as fd:
            metadata = DatasetMetadata.model_validate_json(fd.read())
        return metadata

    @classmethod
    def from_directory(cls, directory: DatasetDirectory) -> "Dataset":
        return cls(**directory.model_dump())
