from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class TaskAdapter(BaseModel, ABC):
    """
    Base class for task adapters.

    Take a dataset and adapt it to the task at hand.
    """

    kind: str

    dataset: AnnotatedDataMatrix

    batch_groupby_cols: list[str] = Field(default_factory=list)
    perturbation_groupby_cols: list[str] = Field(default_factory=list)
    context_groupby_cols: list[str] = Field(default_factory=list)

    @field_validator("batch_groupby_cols", "perturbation_groupby_cols", "context_groupby_cols")
    def validate_groupby_cols_unique(cls, v: list[str]) -> list[str]:
        """
        Assert the groupby cols are unique.
        """
        if len(v) != len(set(v)):
            raise ValueError(f"The groupby cols are not unique: {v}")
        return v

    @model_validator(mode="after")
    def validate_context_groupby_cols(self) -> "TaskAdapter":
        """
        Assert the context groupby cols are a subset of the biological context.
        """
        if not self.dataset.metadata.biological_context <= set(self.context_groupby_cols):
            raise ValueError("The context groupby cols are not a subset of the biological context:")
        return self

    @model_validator(mode="after")
    def validate_no_overlap_between_groupby_cols(self) -> "TaskAdapter":
        """
        Assert there is no overlap between the groupby cols.
        """

        # Assert union is the same length as the sum of the individual groupby cols
        n_union = len(
            set(self.perturbation_groupby_cols + self.context_groupby_cols + self.batch_groupby_cols)
        )
        n_sum = (
            len(self.perturbation_groupby_cols)
            + len(self.context_groupby_cols)
            + len(self.batch_groupby_cols)
        )
        if n_union != n_sum:
            raise ValueError(
                f"The groupby cols overlap: Length of union ({n_union}) != Sum of lengths ({n_sum})"
            )
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
