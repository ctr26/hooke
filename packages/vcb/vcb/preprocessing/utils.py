import numpy as np
import polars as pl


def right_skew(data: np.ndarray):
    d = data.flatten()
    d = d[d != 0]
    return d.mean() / np.median(d)


def integer_only(data: np.ndarray):
    return np.allclose(data, np.round(data), rtol=1e-3)


def summarize_distribution(data: np.ndarray) -> np.ndarray:
    return np.array([data.min(), data.max(), data.mean(), np.median(data), integer_only(data)])


def distribution_summary_similarity(
    summary: np.ndarray, reference: np.ndarray, epsilon: float = 1e-4
) -> float:
    """
    Simple way to compare two distributions based on their summary statistics.
    """
    delta = summary - reference
    pdelta = delta / (np.abs(reference) + epsilon)
    return np.mean(np.abs(pdelta))


def summaries_to_table(summaries: dict[str, np.ndarray]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Metric": ["min", "max", "mean", "median", "integer_only"],
            **{k: v for k, v in summaries.items()},
        },
        schema={"Metric": pl.Utf8, "Before": pl.Float64, "After": pl.Float64},
    )
