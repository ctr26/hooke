from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger
from pydantic import BaseModel, ConfigDict, computed_field, field_validator


class PredictionsPaths(BaseModel):
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


class Predictions(BaseModel):
    """
    A predictions object.

    TODO (cwognum): For future reference: Predictions and Dataset have a lot of similarities.
        They could share a super class, or maybe even be merged into a single class.

    TODO (cwognum): We'll likely want to distinguish different dataset types, e.g. raw counts, embeddings, etc.
        One clear example is the gene_id_column attribute, which is only needed for raw counts.
    """

    paths: PredictionsPaths

    gene_id_column: str = "ensembl_gene_id"
    features_array_name: str | None = None

    _gene_labels_subset: set[str] | None = None
    _cached_features: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("paths")
    def validate_paths(cls, v: PredictionsPaths) -> PredictionsPaths:
        if not v.obs_path.exists():
            raise ValueError(f"obs path {v.obs_path} does not exist")
        if not v.features_path.exists():
            raise ValueError(f"feature path {v.features_path} does not exist")
        return v

    @property
    def obs(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.obs_path)

    @property
    def var(self) -> pl.DataFrame:
        return pl.read_parquet(self.paths.var_path)

    @property
    def gene_labels(self) -> pl.DataFrame:
        return self.var[self.gene_id_column].to_list()

    @property
    def X(self) -> zarr.Array:
        data = zarr.open(self.paths.features_path)
        if self.features_array_name is not None:
            data = data[self.features_array_name]

        if not isinstance(data, zarr.Array):
            raise ValueError(
                f"features path {self.paths.features_path} is not a zarr array. Did you forget to set --features-array-name?"
            )

        if self._cached_features is None:
            logger.info(f"Loading {self.paths.features_path} into memory.")
            self._cached_features = data[:, self._get_gene_mask()]

        return self._cached_features

    def set_gene_labels_subset(self, gene_labels: set[str]) -> None:
        self._gene_labels_subset = gene_labels

        # Also invalidate cached features
        self._cached_features = None

    def _get_gene_mask(self) -> np.ndarray:
        if self._gene_labels_subset is None:
            return np.ones(len(self.gene_labels), dtype=bool)
        return np.isin(self.gene_labels, np.array(list(self._gene_labels_subset)))
