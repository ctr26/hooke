from pathlib import Path

import numpy as np
import polars as pl
import zarr
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator


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

    var_path: Path
    obs_path: Path
    features_path: Path

    features_layer: str | None = None

    _cached_features: np.ndarray | None = None
    _var_mask: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("var_path", "obs_path", "features_path")
    def validate_path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"{v} does not exist")
        return v

    @property
    def obs(self) -> pl.DataFrame:
        return pl.read_parquet(self.obs_path)

    @property
    def var(self) -> pl.DataFrame:
        return pl.read_parquet(self.var_path)

    @property
    def X(self) -> zarr.Array:
        """
        Returns the features.

        Let's make this more robust once data actually no longer fits in memory.
        """
        if self._cached_features is None:
            logger.info(
                f"Loading {self.features_path} into memory. This may take a while."
            )

            # Load the Zarr archive, can be a group or an array
            X = zarr.open(self.features_path)

            if isinstance(X, zarr.Group):
                if self.features_layer is None:
                    raise ValueError(
                        "features_layer is not set, and the features_path is a Zarr group. "
                        "Please set features_layer to the name of the array to load the features from."
                    )
                X = X[self.features_layer]

            # Load the features from the Zarr file to a NumPy array This assumes all features fit in memory.
            # This may not always be the case, and defeats the purpose of using Zarr in the first place,
            # but we'll cross that bridge when we get to it.
            self._cached_features = X[:, self._var_mask]

        return self._cached_features

    @X.setter
    def X(self, features: np.ndarray) -> None:
        self._cached_features = features

    def set_var_mask(self, mask: np.ndarray) -> None:
        """
        Mask the features along the var dimension.
        Also invalidates the cached features, if any.
        """
        self._var_mask = mask
        self._cached_features = None
