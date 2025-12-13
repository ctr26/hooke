from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, field_validator, model_validator

from vcb.data_models.dataset.metadata import DatasetMetadata


class AnnotatedDataMatrix(BaseModel):
    """
    An annotated data matrix.

    An AnnData inspired format, but extended to also support phenomics data.

    Attributes:
        var_path: Parquet file containing the variables.
        obs_path: Parquet file containing the observations.
        features_path: Zarr file containing the features.
        features_layer: If the features are stored as a Zarr group,
            this specifies the name of the array to load the features from.
    """

    obs_path: Path
    features_path: Path
    var_path: Path
    metadata_path: Path | None = None

    features_layer: str | None = None
    zarr_index_column: str | None = None

    # parameterizing mostly for legibility
    _var_dim: int | None = None

    # flag to throw explicit error on potential double call to prepare in TaskAdapter
    # that uses AnnotatedDataMatrix dataset
    _obs_is_prepared: bool = False

    _cached_features: np.ndarray | None = PrivateAttr(default=None)
    _cached_var: pl.DataFrame | None = PrivateAttr(default=None)
    _cached_obs: pl.DataFrame | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Loads in data from assigned paths."""
        obs = pl.read_parquet(self.obs_path)
        var = pl.read_parquet(self.var_path)
        X = self.load_x()
        self._var_dim = X.ndim - 1

        # file corresponding to each row in the obs file
        # the zarr_index_column is necessary as writing is parallized and order not guaranteed for predictions
        if self.zarr_index_column is not None:
            obs_indices = obs[self.zarr_index_column].to_numpy()
            X = X[obs_indices]

        # set obs, var, and x to class instance
        self.update(obs=obs, var=var, X=X)

    @field_validator("var_path", "obs_path", "features_path")
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"{v} does not exist")
        return v

    @field_validator("metadata_path")
    def validate_optional_path_exists(cls, v: Path) -> Path:
        if v is not None and not v.exists():
            raise ValueError(f"{v} does not exist")
        return v

    @model_validator(mode="after")
    def validate_features_layer(cls, m: "AnnotatedDataMatrix") -> "AnnotatedDataMatrix":
        root = zarr.open(m.features_path, mode="r")

        if m.features_layer is not None and isinstance(root, zarr.Array):
            logger.warning(
                f"Ignoring features_layer `{m.features_layer}` because {m.features_path} is not a Zarr Group."
            )

        if isinstance(root, zarr.Group):
            array_keys = list(root.array_keys())
            if m.features_layer is None:
                raise ValueError(
                    "features_layer needs to be set when the features_path is a Zarr Group. "
                    f"Set it to one of the following arrays: {array_keys}"
                )
            elif m.features_layer not in array_keys:
                raise ValueError(
                    f"features_layer {m.features_layer} not found in {m.features_path}. "
                    f"These are the available arrays: {array_keys}"
                )

        columns = pl.read_parquet(m.obs_path).columns
        if m.zarr_index_column is not None and m.zarr_index_column not in columns:
            raise ValueError(
                f"zarr_index_column {m.zarr_index_column} not found in {m.obs_path}. "
                f"These are the available columns: {columns}"
            )
        return m

    @property
    def obs(self) -> pl.DataFrame:
        return self._cached_obs

    @property
    def var(self) -> pl.DataFrame:
        return self._cached_var

    @property
    def X(self) -> zarr.Array:
        return self._cached_features

    @X.setter
    def X(self, features: np.ndarray) -> None:
        self.update(X=features)

    @obs.setter
    def obs(self, obs: pl.DataFrame) -> None:
        self.update(obs=obs)

    @var.setter
    def var(self, var: pl.DataFrame) -> None:
        self.update(var=var)

    def update(self, obs=None, var=None, X=None) -> None:
        """
        primary setter for anndata objects, including dimension match checks

        Args:
            obs (polars.DataFrame): The new observations, None to keep existing.
            var (polars.DataFrame): The new variables, None to keep existing.
            X (np.ndarray): The new features, None to keep existing.
        """
        # determine the resulting dimensions if this is executed
        new_obs_len = obs.shape[0] if obs is not None else self.obs.shape[0]
        new_var_len = var.shape[0] if var is not None else self.var.shape[0]
        new_x_shape = X.shape if X is not None else self.X.shape

        # if dimensions are valid, set the new values
        if len(new_x_shape) - 1 != self._var_dim:
            raise ValueError(
                f"update to X would change dimensions from {self._var_dim + 1} to {len(new_x_shape)}"
            )

        if (new_obs_len, new_var_len) == (new_x_shape[0], new_x_shape[self._var_dim]):
            if obs is not None:
                self._cached_obs = obs
            if var is not None:
                self._cached_var = var
            if X is not None:
                self._cached_features = X
        else:
            # else, report dimension mismatch, and hint how to set when changing a dimension
            raise ValueError(
                f"invalid shape for anndata: obs = {new_obs_len}, var = {new_var_len}, "
                f"but (obs, var) from X = {new_x_shape}; "
                "use `update` method directly to set contingent elements of obs, var, X simultaneously"
            )

    def filter(self, obs_indices=None, var_indices=None):
        """filters whole anndata along obs and/or var dimensions"""
        X = self.X

        if obs_indices is not None:
            obs = self.obs[obs_indices]
            X = X[obs_indices]
        else:
            obs = None

        if var_indices is not None:
            var = self.var[var_indices]
            # var dim is generally the last, only 1 or 2 for foreseeable future
            if self._var_dim != 1:
                raise NotImplementedError("X dimensions != 2 are not currently implemented")

            X = X[:, var_indices]
        else:
            var = None

        self.update(X=X, obs=obs, var=var)

    @computed_field
    @property
    def metadata(self) -> DatasetMetadata | None:
        if self.metadata_path is None:
            return None
        with open(self.metadata_path, "r") as fd:
            metadata = DatasetMetadata.model_validate_json(fd.read())
        return metadata

    @property
    def dataset_id(self) -> str | None:
        if self.metadata is None:
            return None
        return self.metadata.dataset_id

    def load_x(self):
        """Reads X features from zarr file, collapsing middle dimension if applicable"""
        logger.info(f"Loading {self.features_path} into memory. This may take a while.")

        # Load the Zarr archive, can be a group or an array
        X = zarr.open(self.features_path)

        if isinstance(X, zarr.Group):
            if self.features_layer is None:
                raise ValueError(
                    "features_layer is not set, and the features_path is a Zarr group. "
                    "Please set features_layer to the name of the array to load the features from."
                )
            X = X[self.features_layer]

        X = X[:]  # actual, slow read in

        # collapse patch embeddings if present
        # assumes last two dimensions are the patch and embedding dimensions
        if X.ndim == 3:
            old_shape = X.shape
            X = X.mean(axis=-2)
            logger.warning(f"Collapsing patch embeddings. Input shape: {old_shape}, output shape: {X.shape}")

        elif X.ndim != 2:  # 2 is expected, with no action required, everything else unexpected
            raise ValueError(f"Unexpected shape {X.shape} does not have length in [2, 3]")

        return X


class TxAnnotatedDataMatrix(AnnotatedDataMatrix):
    """
    An annotated data matrix for transcriptomics data.
    """

    var_gene_id_column: str = "ensembl_gene_id"

    @property
    def gene_ids(self) -> list[str]:
        return self.var[self.var_gene_id_column].to_list()
