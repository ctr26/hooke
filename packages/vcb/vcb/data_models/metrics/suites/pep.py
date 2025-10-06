from typing import Literal

import polars as pl
from pydantic import field_validator

from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.metrics.utils import manual_group_by


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

    def evaluate(self) -> pl.DataFrame:
        rows = []

        # Groupby context
        for context in manual_group_by(self.predictions.dataset.obs, self.predictions.context_groupby_cols):
            context_predicate = [pl.col(col) == value for col, value in context.items()]
            context_obs = self.predictions.dataset.obs.filter(*context_predicate)

            # Groupby metric
            for group in manual_group_by(context_obs, self.predictions.perturbation_groupby_cols):
                metric_predicate = [pl.col(col) == value for col, value in group.items()]
                metric_predicate = context_predicate + metric_predicate

                # Get all the data we need from the dataset
                y_base = self.ground_truth.get_basal_states(*context_predicate)
                y_true = self.ground_truth.get_perturbed_states(*metric_predicate)
                y_pred = self.predictions.get_perturbed_states(*metric_predicate)

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
                    rows.append({"score": score, "metric": label, **context, **group})

        return pl.DataFrame(rows)
