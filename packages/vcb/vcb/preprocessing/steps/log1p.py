from typing import Literal

import numpy as np

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.base import PreprocessingStep


class Log1pStep(PreprocessingStep):
    """
    A step that will log1p transform the data.
    """

    kind: Literal["log1p"] = "log1p"

    transform_ground_truth: bool = True
    transform_predictions: bool = True

    def transform(
        self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix
    ) -> tuple[AnnotatedDataMatrix, AnnotatedDataMatrix]:
        if self.transform_ground_truth:
            ground_truth.X = np.log1p(ground_truth.X)
        if self.transform_predictions:
            predictions.X = np.log1p(predictions.X)
        return ground_truth, predictions


class InverseLog1pStep(PreprocessingStep):
    """
    A step that will inverse log1p transform the data.
    """

    kind: Literal["inverse_log1p"] = "inverse_log1p"

    transform_ground_truth: bool = False
    transform_predictions: bool = True

    def transform(
        self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix
    ) -> tuple[AnnotatedDataMatrix, AnnotatedDataMatrix]:
        if self.transform_ground_truth:
            ground_truth.X = np.exp(ground_truth.X) - 1
        if self.transform_predictions:
            predictions.X = np.exp(predictions.X) - 1
        return ground_truth, predictions
