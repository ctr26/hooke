"""Utils for fast FD (DinoV2) evaluation.
This provides a fast and reliable signal, which is easy to use during
training loops. It should match results from dgm-eval extremely closely [1].

For FID comparisons with other methods on Imagenet, save a batch of generated
images and use the standard OpenAI ADM evaluator script [2].

[1] https://github.com/layer6ai-labs/dgm-eval
[2] https://github.com/openai/guided-diffusion/tree/main/evaluations
"""

import numpy as np
import sklearn.metrics
import torch
from scipy import linalg

def compute_statistics(reps):
    mu = np.mean(reps, axis=0)
    sigma = np.cov(reps, rowvar=False)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def compute_cossim(reps1, reps2):
    mu1 = np.mean(reps1, axis=0)
    mu2 = np.mean(reps2, axis=0)
    return np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2))


def compute_fd(reps1, reps2):
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)
    sqrt_trace = np.real(linalg.eigvals(sigma1 @ sigma2) ** 0.5).sum()  # type: ignore
    return ((mu1 - mu2) ** 2).sum() + sigma1.trace() + sigma2.trace() - 2 * sqrt_trace


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    d = dict(precision=precision, recall=recall, density=density, coverage=coverage)
    return d
