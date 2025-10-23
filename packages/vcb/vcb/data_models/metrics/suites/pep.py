from typing import Literal

import polars as pl
from pydantic import field_validator

from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.utils import predicate_group_by


class PerturbationEffectPredictionSuite(MetricSuite):
    """
    Perturbation effect prediction metric suites.
    """

    kind: Literal["perturbation_effect_prediction"] = "perturbation_effect_prediction"

    @field_validator("metric_labels")
    @classmethod
    def validate_metrics_allowlist(cls, v: set[str]) -> set[str]:
        """
        Assert all metrics are in the metric_labels.
        """
        allowlist = ["mse", "pearson", "cosine", "pearson_delta", "cosine_delta"]
        for metric in v:
            if metric not in allowlist:
                raise ValueError(f"Metric {metric} is not supported for perturbation effect prediction tasks")
        return v

    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        rows = []

        # Groupby context and batch
        groupby_cols = predictions.batch_groupby_cols + predictions.context_groupby_cols

        for batch_context, batch_context_obs, batch_context_predicate in predicate_group_by(
            predictions.dataset.obs,
            groupby_cols,
            # NOTE (cwognum): This may seem like the wrong description, but it's actually correct.
            # I place this description here so that it's visible in the top-level progress bar.
            description=f"Computing {self.kind} suite per {predictions.perturbation_groupby_cols}",
        ):
            # Groupby metric
            for perturbation, _, perturbation_predicate in predicate_group_by(
                batch_context_obs, predictions.perturbation_groupby_cols
            ):
                metric_predicate = batch_context_predicate + perturbation_predicate

                # Get all the data we need from the dataset
                y_base = ground_truth.get_basal_states(*batch_context_predicate)
                y_true = ground_truth.get_perturbed_states(*metric_predicate)
                y_pred = predictions.get_perturbed_states(*metric_predicate)

                for label, metric in self.metrics.items():
                    # Skip distributional metrics if we don't want to use them
                    if metric.is_distributional and not self.use_distributional_metrics:
                        continue

                    # Prepare arguments
                    kwargs = {"y_pred": y_pred, "y_true": y_true, **metric.kwargs}
                    if metric.is_delta_metric:
                        kwargs["y_base"] = y_base

                    if not metric.is_distributional:
                        for key in ["y_pred", "y_true", "y_base"]:
                            if key in kwargs:
                                kwargs[key] = kwargs[key].mean(axis=0)

                    # Compute score
                    score = metric.fn(**kwargs)
                    rows.append({"score": score, "metric": label, **batch_context, **perturbation})

        return pl.DataFrame(rows)
