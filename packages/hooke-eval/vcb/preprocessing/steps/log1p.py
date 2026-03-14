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

    def transform_single(self, data: AnnotatedDataMatrix):
        data.X = np.log1p(data.X)

    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        if self.transform_ground_truth:
            self.transform_single(ground_truth)
        if self.transform_predictions:
            self.transform_single(predictions)


class InverseLog1pStep(PreprocessingStep):
    """
    A step that will inverse log1p transform the data.
    """

    kind: Literal["inverse_log1p"] = "inverse_log1p"

    transform_ground_truth: bool = False
    transform_predictions: bool = True

    def transform_single(self, data: AnnotatedDataMatrix):
        data.X = np.exp(data.X) - 1

    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        if self.transform_ground_truth:
            self.transform_single(ground_truth)
        if self.transform_predictions:
            self.transform_single(predictions)
