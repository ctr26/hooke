from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from pydantic import BaseModel

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class TaskAdapter(BaseModel, ABC):
    """
    Base class for task adapters.

    Take a dataset and adapt it to the task at hand.
    """

    kind: str

    dataset: AnnotatedDataMatrix

    def prepare(self) -> None:
        pass

    @abstractmethod
    def get_basal_states(self, *predictates: pl.Expr) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_basal_obs(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_perturbations(self, *predictates: pl.Expr) -> np.ndarray:
        pass

    @abstractmethod
    def get_perturbed_states(self, *predictates: pl.Expr) -> np.ndarray:
        pass

    @abstractmethod
    def get_all_perturbed_obs(self) -> pl.DataFrame:
        pass

    def get_biological_context(self, *predictates: pl.Expr) -> dict:
        if self.dataset.metadata is None:
            raise RuntimeError("Can't get biological context if metadata is not set")
        obs = self.dataset.obs.filter(*predictates)
        return {col: obs[col] for col in self.dataset.metadata.biological_context}
