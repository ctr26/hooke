from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.task.base import TaskAdapter


class BaseBaseline(BaseModel, ABC):
    """
    Base class for all baselines.
    """

    @abstractmethod
    def fit(self, task: TaskAdapter) -> Self:
        pass

    @abstractmethod
    def predict(self, task: TaskAdapter) -> AnnotatedDataMatrix:
        pass
