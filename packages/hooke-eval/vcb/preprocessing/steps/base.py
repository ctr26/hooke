from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class PreprocessingStep(BaseModel, ABC):
    """
    A preprocessing step.
    """

    kind: str

    def fit(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix) -> Self:
        """
        Fit the preprocessing step.
        """
        return self

    @abstractmethod
    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        """
        Transform the preprocessing step.
        """
        raise NotImplementedError
