"""Preprocessing components for Hooke pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Preprocessor(ABC):
    """Abstract base for preprocessing steps."""

    @abstractmethod
    def fit(self, X: NDArray[np.floating[Any]]) -> Preprocessor:
        """Fit the preprocessor to data."""

    @abstractmethod
    def transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Transform data."""

    def fit_transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return self.fit(X).transform(X)


class StandardScaler(Preprocessor):
    """Zero-mean, unit-variance scaling."""

    def __init__(self) -> None:
        self._mean: NDArray[np.floating[Any]] | None = None
        self._std: NDArray[np.floating[Any]] | None = None

    def fit(self, X: NDArray[np.floating[Any]]) -> StandardScaler:
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if self._mean is None or self._std is None:
            raise RuntimeError("StandardScaler not fitted. Call fit() first.")
        return (X - self._mean) / self._std


class LabelEncoder(Preprocessor):
    """Encode string labels as integers."""

    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}

    def fit(self, X: NDArray[np.floating[Any]]) -> LabelEncoder:
        unique = sorted(set(X.ravel().tolist()))
        self._mapping = {str(v): i for i, v in enumerate(unique)}
        return self

    def transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if not self._mapping:
            raise RuntimeError("LabelEncoder not fitted. Call fit() first.")
        result = np.array([self._mapping[str(v)] for v in X.ravel()], dtype=np.float64)
        return result.reshape(X.shape)


class PreprocessingPipeline:
    """Composable chain of preprocessing steps."""

    def __init__(self, steps: list[Preprocessor] | None = None) -> None:
        self.steps: list[Preprocessor] = steps or []

    def fit(self, X: NDArray[np.floating[Any]]) -> PreprocessingPipeline:
        current = X
        for step in self.steps:
            current = step.fit_transform(current)
        return self

    def transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        current = X
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return self.fit(X).transform(X)
