from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
)


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

    TODO (cwognum): For future reference: Predictions and Dataset have a lot of similarities.
        They could share a super class, or maybe even be merged into a single class.

    TODO (cwognum): We'll likely want to distinguish different dataset types, e.g. raw counts, embeddings, etc.
        One clear example is the gene_id_column attribute, which is only needed for raw counts.
    """

    paths: DatasetPaths

    gene_id_column: str = "ensembl_gene_id"

    _gene_labels_subset: set[str] | None = None
    _cached_features: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    def var(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.var_path)

    @property
    def X(self) -> zarr.Array:
        """
        Returns the features.

        Supports filtering by gene labels and loads the features into memory.
        The upfront cost of loading everything into memory is high, but it speeds things up downstream.

        Let's make this more robust once data actually no longer fits in memory.
        """
        if self._cached_features is None:
            logger.info(f"Loading {self.paths.features_path} into memory.")
            arr = zarr.open(self.paths.features_path)
            self._cached_features = arr[:, self._get_gene_mask()]
        return self._cached_features

    @computed_field
    @property
    def metadata(self) -> DatasetMetadata:
        with open(self.paths.metadata_path, "r") as fd:
            metadata = DatasetMetadata.model_validate_json(fd.read())
        return metadata

    @property
    def gene_labels(self) -> pl.DataFrame:
        return self.var[self.gene_id_column].to_list()

    def set_gene_labels_subset(self, gene_labels: set[str]) -> None:
        self._gene_labels_subset = gene_labels

        # Also invalidate cached features
        self._cached_features = None

    def _get_gene_mask(self) -> np.ndarray:
        if self._gene_labels_subset is None:
            return np.ones(len(self.gene_labels), dtype=bool)
        gene_mask = np.isin(self.gene_labels, np.array(list(self._gene_labels_subset)))

        # Temporary fix: Deduplicate the gene labels.
        # Because we're working with Ensembl IDs, this shouldn't be needed.
        dedup_mask = np.zeros(len(self.gene_labels), dtype=bool)
        deduplicate_indices = self.var[self.gene_id_column].arg_unique().to_numpy()
        dedup_mask[deduplicate_indices] = True
        gene_mask = gene_mask & dedup_mask

        return gene_mask
