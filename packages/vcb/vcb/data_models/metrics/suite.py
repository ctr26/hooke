from abc import ABC, abstractmethod
from typing import Callable

import polars as pl
from pydantic import BaseModel, Field, model_validator

from vcb.data_models.task.base import TaskAdapter
from vcb.metrics import METRICS


class MetricSuite(BaseModel, ABC):
    """
    Base class for metric suites.
    """

    kind: str

    metric_labels: set[str] = Field(default_factory=set)
    use_distributional_metrics: bool = True

    context_groupby_cols: set[str] = Field(default_factory=set)
    perturbation_groupby_cols: set[str] | None = None

    @model_validator(mode="after")
    def validate_no_overlap_between_groupby_cols(self) -> "MetricSuite":
        """
        Assert there is no overlap between the perturbation and context groupby cols.

        We assume the perturbation_groupby_cols are a superset of the context_groupby_cols,
        but expect the user to only specify the difference.
        """
        if self.perturbation_groupby_cols is None:
            return self

        intersection = self.perturbation_groupby_cols & self.context_groupby_cols
        if intersection:
            raise ValueError(f"Perturbation groupby cols and context groupby cols overlap: {intersection}")
        return self

    @property
    def metrics(self) -> dict[str, Callable]:
        return {metric: METRICS[metric] for metric in self.metric_labels}

    def prepare(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> None:
        ground_truth.prepare()
        predictions.prepare()

    @abstractmethod
    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        pass
