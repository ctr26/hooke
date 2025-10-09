from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, model_validator

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class TaskAdapter(BaseModel, ABC):
    """
    Base class for task adapters.

    Take a dataset and adapt it to the task at hand.
    """

    dataset: AnnotatedDataMatrix

    context_groupby_cols: set[str] = Field(default_factory=set)
    perturbation_groupby_cols: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def validate_biological_context_in_groupby_cols(self) -> "TaskAdapter":
        """
        Assert the biological context is in the groupby cols.
        """
        if not self.dataset.metadata.biological_context <= self.context_groupby_cols:
            raise ValueError("Biological context is not in the groupby cols")
        return self

    @model_validator(mode="after")
    def validate_no_overlap_between_groupby_cols(self) -> "TaskAdapter":
        """
        Assert there is no overlap between the perturbation and context groupby cols.

        We assume the perturbation_groupby_cols are a superset of the context_groupby_cols,
        but expect the user to only specify the difference.
        """
        intersection = self.perturbation_groupby_cols & self.context_groupby_cols
        if intersection:
            raise ValueError(f"Perturbation groupby cols and context groupby cols overlap: {intersection}")
        return self

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
