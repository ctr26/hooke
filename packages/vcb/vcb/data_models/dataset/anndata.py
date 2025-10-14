import uuid
from pathlib import Path

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
    var_path: Path | None = None
    metadata_path: Path | None = None

    features_layer: str | None = None
    zarr_index_column: str | None = None

    _cached_features: np.ndarray | None = PrivateAttr(default=None)
    _cached_obs: pl.DataFrame | None = None
    _var_indices: np.ndarray | None = None
    _obs_indices: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("var_path", "obs_path", "features_path", "metadata_path")
    def validate_path_exists(cls, v: Path) -> Path:
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
        if self._cached_obs is None:
            obs = pl.read_parquet(self.obs_path)

            # If specified, filter down the observations.
            if self._obs_indices is not None:
                # To make minimal assumptions about columns in the obs,
                # we add a temporary, randomly named row index we remove after filtering.
                tmp_column = uuid.uuid4().hex
                obs = obs.with_row_index(tmp_column)
                obs = obs.filter(pl.col(tmp_column).is_in(self._obs_indices))
                obs = obs.drop(tmp_column)

            self._cached_obs = obs
        return self._cached_obs

    @property
    def var(self) -> pl.DataFrame:
        if self.var_path is not None:
            return pl.read_parquet(self.var_path)

    @property
    def X(self) -> zarr.Array:
        """
        Returns the features.

        Let's make this more robust once data actually no longer fits in memory.
        """
        if self._cached_features is None:
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

            if self.zarr_index_column is not None:
                # For predictions, we need to explicitly match observations to features.
                # We do this using the zarr_index_column to reorder the features.
                # We do not need to worry about the `_obs_indices` here, because `self.obs` is already filtered.
                obs_indices = self.obs[self.zarr_index_column].to_numpy()
            elif self._obs_indices is not None:
                # Otherwise, if `_obs_indices` is set, we use it to filter down the features.
                obs_indices = self._obs_indices
            else:
                # Otherwise, we load all features.
                obs_indices = slice(None)

            # Load the features from the Zarr file to a NumPy array This assumes all features fit in memory.
            # This may not always be the case, and defeats the purpose of using Zarr in the first place,
            # but we'll cross that bridge when we get to it.
            if self._var_indices is not None:
                self._cached_features = X.oindex[obs_indices, self._var_indices]
            else:
                self._cached_features = X.oindex[obs_indices, :]

        # collapse patch embeddings if present
        # assumes last two dimensions are the patch and embedding dimensions
        if len(self._cached_features.shape) > 2:
            logger.warning(f"Collapsing patch embeddings. Input shape: {self._cached_features.shape}")
            self._cached_features = self._cached_features.mean(axis=-2)

        return self._cached_features

    @X.setter
    def X(self, features: np.ndarray) -> None:
        self._cached_features = features

    @obs.setter
    def obs(self, obs: pl.DataFrame) -> None:
        self._cached_obs = obs

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

    def invalidate_cache(self) -> None:
        """
        Invalidate the cached features and obs.
        """
        self._cached_obs = None
        self._cached_features = None

    def set_var_indices(self, indices: np.ndarray) -> None:
        """
        Mask the features along the var dimension.
        Also invalidates the cached features, if any.
        """
        self._var_indices = np.sort(indices)
        self.invalidate_cache()

    def set_obs_indices(self, indices: np.ndarray) -> None:
        """
        Mask the features along the obs dimension.
        Also invalidates the cached features, if any.
        """
        self._obs_indices = np.sort(indices)
        self.invalidate_cache()
