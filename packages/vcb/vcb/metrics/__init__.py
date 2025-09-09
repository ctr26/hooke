import numpy as np
from scipy.stats import pearsonr

# from vcb.metrics.distributional.fd import compute_efficient_FD_with_reps
from vcb.metrics.distributional.mmd import compute_e_distance

DISTRIBUTIONAL_METRICS = {
    # "fd_efficient": compute_efficient_FD_with_reps,
    "mmd": compute_e_distance,
}

SAMPLE_METRICS = {
    "mse": lambda x, y, b: np.mean((x - y) ** 2),
    "pearson": lambda x, y, b: pearsonr(x, y)[0],
    "cosine": lambda x, y, b: np.mean(
        np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    ),
    "cosine_delta": lambda x, y, b: np.mean(
        np.dot(x - b, y - b) / (np.linalg.norm(x - b) * np.linalg.norm(y - b))
    ),
    "pearson_delta": lambda x, y, b: pearsonr(x - b, y - b)[0],
}
