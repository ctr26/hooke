"""Metric primitives for evaluating generative models.

Includes Frechet Distance variants, cosine similarity, PRDC, energy distance,
and retrieval metrics. Consolidates utils/evaluation.py and hooke-predict metrics.
"""

import numpy as np
import sklearn.metrics
from scipy import linalg

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Frechet Distance
# ---------------------------------------------------------------------------


def compute_fd(reps1, reps2):
    """Fast FD using eigenvalue decomposition (no sqrtm)."""
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)
    sqrt_trace = np.real(linalg.eigvals(sigma1 @ sigma2) ** 0.5).sum()  # type: ignore
    return ((mu1 - mu2) ** 2).sum() + sigma1.trace() + sigma2.trace() - 2 * sqrt_trace


def compute_fd_sqrtm(reps1, reps2, eps=1e-6):
    """FD using scipy sqrtm (standard formulation, slower but reference-grade)."""
    mu1, sigma1 = compute_statistics(reps1)
    mu2, sigma2 = compute_statistics(reps2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def compute_fd_infinity(reps1, reps2, num_points=15):
    """FD extrapolated to infinite sample size via linear fit on 1/N."""
    mu1, sigma1 = compute_statistics(reps1)

    fd_batches = np.linspace(min(5000, max(len(reps2) // 10, 2)), len(reps2), num_points).astype("int32")

    rng = np.random.default_rng()
    fds = np.array(
        [
            compute_fd_sqrtm_from_stats(mu1, sigma1, *compute_statistics(rng.choice(reps2, n, replace=False)))
            for n in fd_batches
        ]
    )

    # Linear fit: FD(N) = FD_inf + slope / N  =>  polyfit on (1/N, FD)
    coeffs = np.polyfit(1.0 / fd_batches, fds, deg=1)
    return float(np.polyval(coeffs, 0.0))


def compute_fd_sqrtm_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FD from pre-computed statistics (sqrtm variant)."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ---------------------------------------------------------------------------
# Energy distance (MMD)
# ---------------------------------------------------------------------------


def compute_e_distance(x, y):
    """Energy distance between two sets of samples."""
    sigma_X = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1).mean()
    sigma_Y = ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1).mean()
    delta = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1).mean()
    return float(2 * delta - sigma_X - sigma_Y)


# ---------------------------------------------------------------------------
# PRDC (Precision, Recall, Density, Coverage)
# ---------------------------------------------------------------------------


def compute_pairwise_distance(data_x, data_y=None):
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()

    recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    d = dict(precision=precision, recall=recall, density=density, coverage=coverage)
    return d


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def compute_retrieval(distance_matrix):
    """Normalized retrieval score from a pairwise distance matrix.

    The diagonal entry (i, i) is the distance between the i-th generated sample
    and its ground-truth counterpart.  We rank each row and compute how often the
    correct match ranks first.

    Returns a score in [0, 1] where 1 is perfect retrieval and 0.5 is random.
    """
    ranks_argsort = np.argsort(distance_matrix, axis=1)
    ranks_indicator = ranks_argsort == np.arange(ranks_argsort.shape[0]).reshape(-1, 1)
    ranks = np.nonzero(ranks_indicator)[1]
    return float(1 - np.mean(ranks) / (distance_matrix.shape[0] - 1))
