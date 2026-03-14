from typing import Callable

from pydantic import BaseModel, Field


class MinimalMetricInfo(BaseModel):
    """
    Simple metric metadata

    Attributes:
        fn: The callable that actually computes the metric.
        kwargs: Additional parameters required for the metric.
    """

    fn: Callable
    kwargs: dict = Field(default_factory=dict)
