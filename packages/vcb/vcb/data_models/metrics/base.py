from typing import Callable, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Metric: TypeAlias = Literal[
    "mse",
    "pearson",
    "cosine",
    "pearson_delta",
    "cosine_delta",
    "edistance",
    "retrieval_mae",
    "retrieval_edistance",
]


class MetricInfo(BaseModel):
    """
    Metric metadata

    Attributes:
        fn: The callable that actually computes the metric.
        kwargs: Additional parameters required for the metric.
    """

    fn: Callable
    kwargs: dict = Field(default_factory=dict)

    is_distributional: bool = False
    is_delta_metric: bool = False

    model_config = ConfigDict(frozen=True)
