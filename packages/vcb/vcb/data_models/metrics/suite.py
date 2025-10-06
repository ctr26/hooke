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

    ground_truth: TaskAdapter
    predictions: TaskAdapter

    metric_labels: set[str] = Field(default_factory=set)
    use_distributional_metrics: bool = True

    @model_validator(mode="after")
    def validate_consistency_between_tasks(self) -> "MetricSuite":
        """
        Assert the ground truth and predictions are the same type.
        """
        if not isinstance(self.ground_truth, type(self.predictions)):
            raise ValueError("Ground truth and predictions must be the same type")
        if self.ground_truth.context_groupby_cols != self.predictions.context_groupby_cols:
            raise ValueError("Ground truth and predictions do not have the same context groupby cols")
        if self.ground_truth.perturbation_groupby_cols != self.predictions.perturbation_groupby_cols:
            raise ValueError("Ground truth and predictions do not have the same perturbation groupby cols")
        return self

    @property
    def metrics(self) -> dict[str, Callable]:
        return {metric: METRICS[metric] for metric in self.metric_labels}

    def prepare(self) -> None:
        self.ground_truth.prepare()
        self.predictions.prepare()

    @abstractmethod
    def evaluate(self) -> pl.DataFrame:
        pass
