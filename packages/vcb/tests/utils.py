import numpy as np
import polars as pl


def mock_transcriptomics_var(n: int):
    return pl.DataFrame({"ensembl_gene_id": [f"ENSG{i:012d}" for i in range(n)]})


def assert_perfect_performance(
    results: pl.DataFrame,
    correlation_metrics: list[str],
    retrieval_metrics: list[str],
    error_metrics: list[str],
):
    summary = (
        results.group_by("metric")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("metric")
    )

    # Correlation metrics and Retrieval metrics need to be maximized
    for metric in correlation_metrics + retrieval_metrics:
        assert np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 1.0)

    # Error metrics need to be minimized
    for metric in error_metrics:
        assert np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 0.0)


def assert_imperfect_performance(
    results: pl.DataFrame,
    correlation_metrics: list[str],
    retrieval_metrics: list[str],
    error_metrics: list[str],
):
    summary = (
        results.group_by("metric")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("metric")
    )

    # Correlation metrics and Retrieval metrics need to be maximized
    for metric in correlation_metrics + retrieval_metrics:
        assert not np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 1.0)

    # Error metrics need to be minimized
    for metric in error_metrics:
        assert not np.isclose(summary.filter(pl.col("metric") == metric)["mean"].item(), 0.0)
