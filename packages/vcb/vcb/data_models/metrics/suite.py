from abc import ABC, abstractmethod
from typing import Callable

import polars as pl
from pydantic import BaseModel, Field

from vcb.data_models.task.base import TaskAdapter
from vcb.metrics import METRICS


class MetricSuite(BaseModel, ABC):
    """
    Base class for metric suites.
    """

    kind: str

    metric_labels: set[str] = Field(default_factory=set)
    use_distributional_metrics: bool = True

    @property
    def metrics(self) -> dict[str, Callable]:
        return {metric: METRICS[metric] for metric in self.metric_labels}

    @abstractmethod
    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        pass
