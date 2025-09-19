import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict


class InMemoryPredictions(BaseModel):
    """
    A baseline predictions object.

    Allows baseline predictions to be used for evaluation directly
    without having to save them to file.
    """

    obs: pl.DataFrame
    X: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def obs(self) -> pl.DataFrame:
        return self.obs

    @property
    def X(self) -> np.ndarray:
        return self.X
