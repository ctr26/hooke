"""Preprocessing — demo stub."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Preprocessor(Protocol):
    def fit(self, data: np.ndarray) -> None: ...
    def transform(self, data: np.ndarray) -> np.ndarray: ...
    def fit_transform(self, data: np.ndarray) -> np.ndarray: ...


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        pass

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return data


class LabelEncoder:
    def fit(self, labels: np.ndarray) -> None:
        pass

    def transform(self, labels: np.ndarray) -> np.ndarray:
        return labels

    def fit_transform(self, labels: np.ndarray) -> np.ndarray:
        return labels
