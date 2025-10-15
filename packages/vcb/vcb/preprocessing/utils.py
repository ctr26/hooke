import numpy as np


def mean_relative_abs_error(dist_stats_ref, dist_stats_pred, epsilon=0.0001):
    """MAE normalized to dist_stats_ref

    while the code will work for any array-like dist_stats* arguments, it's impolemented here to be used with distributional stats,
    currently: mean, max, skew, and whether the data is integers only or not
    as a quick & dirty way to check if any requested transformations have us comparing apples to apples (or not)"""
    v1 = np.array(dist_stats_ref)
    v2 = np.array(dist_stats_pred)
    delta = v1 - v2
    pdelta = delta / (np.abs(v1) + epsilon)
    return np.mean(np.abs(pdelta))


def right_skew(data: np.ndarray):
    d = data.flatten()
    d = d[d != 0]
    return d.mean() / np.median(d)


def integer_only(data: np.ndarray):
    return np.allclose(data, np.round(data), rtol=1e-3)
