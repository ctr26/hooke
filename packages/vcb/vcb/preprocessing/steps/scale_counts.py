from typing import Literal, Self

import numpy as np
from loguru import logger
from pydantic import computed_field

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.base import PreprocessingStep


class ScaleCountsStep(PreprocessingStep):
    """
    A step that will scale the counts of a transcriptomics data matrix.
    """

    kind: Literal["scale_counts"] = "scale_counts"

    transform_ground_truth: bool = True
    transform_predictions: bool = False

    library_size: int | None = None
    _desired_library_size: int | None = None

    @computed_field
    @property
    def desired_library_size(self) -> int | None:
        if self.library_size is not None:
            return self.library_size
        return self._desired_library_size

    @desired_library_size.setter
    def desired_library_size(self, value: int | None):
        if value is None:
            raise ValueError("The desired library size cannot be set to None")
        if value <= 0:
            raise ValueError("The desired library size must be positive")
        if not isinstance(value, int):
            raise ValueError("The desired library size must be an integer")
        self._desired_library_size = value

    @property
    def fitted(self) -> bool:
        return self.desired_library_size is not None

    def fit(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix) -> Self:
        """If you don't know the desired library size, you can fit it from a reference array."""

        if self.desired_library_size is not None:
            logger.info(f"Desired library size already set to {self.desired_library_size}. Skipping fit!")
            return self

        self.desired_library_size = int(round(np.median(np.sum(ground_truth.X, axis=1))))
        logger.info(f"Fitted desired library size to {self.desired_library_size} from ground truth")
        return self

    def _transform_single(self, data: AnnotatedDataMatrix) -> AnnotatedDataMatrix:
        """
        Transform a single data matrix to the desired library size.
        """
        d = data.X.copy()
        d = d / np.sum(d, axis=1, keepdims=True)
        d = d * self.desired_library_size
        data.X = d
        return data

    def transform(
        self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix
    ) -> tuple[AnnotatedDataMatrix, AnnotatedDataMatrix]:
        if not self.fitted:
            raise RuntimeError(
                "The desired library size is not set. "
                "Please call fit() first or set the library_size property."
            )

        # Transform the data
        if self.transform_ground_truth:
            ground_truth = self._transform_single(ground_truth)
        if self.transform_predictions:
            predictions = self._transform_single(predictions)

        return ground_truth, predictions
