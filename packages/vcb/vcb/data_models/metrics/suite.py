from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, Field, field_validator

from vcb.data_models.metrics.metric_info import MinimalMetricInfo
from vcb.data_models.task.base import TaskAdapter


class MetricSuite(BaseModel, ABC):
    """
    Base class for metric suites.
    """

    kind: str

    metric_labels: set[str] = Field(default_factory=set)
    _all_supported_metrics: ClassVar[dict[str, MinimalMetricInfo]] = {}

    @property
    def metrics(self) -> dict[str, MinimalMetricInfo]:
        collect = {}
        for label in self.metric_labels:
            info = self._all_supported_metrics[label]
            collect[label] = info
        return collect

    @field_validator("metric_labels")
    @classmethod
    def validate_metrics_allowlist(cls, v: set[str]) -> set[str]:
        """
        Assert all metrics are in the metric_labels.
        """
        for metric in v:
            if metric not in cls._all_supported_metrics:
                raise ValueError(f"Metric {metric} is not supported for phenorescue tasks")
        return v

    @abstractmethod
    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        pass
