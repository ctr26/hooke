"""
(Re)implement simple metrics such that they all have a consistent interface.
"""

import numpy as np
from scipy.stats import pearsonr


def cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cosine similarity between the predicted and ground truth perturbed states."""
    return np.mean(np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between the predicted and ground truth perturbed states."""
    return pearsonr(y_true, y_pred)[0]


def cosine_delta(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    """Cosine similarity between the predicted and ground truth deltas."""
    return cosine(y_true - y_base, y_pred - y_base)


def pearson_delta(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    """Pearson correlation between the predicted and ground truth deltas."""
    return pearsonr(y_true - y_base, y_pred - y_base)[0]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between the predicted and ground truth perturbed states."""
    return np.mean((y_true - y_pred) ** 2)
