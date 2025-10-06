from typing import Literal

import polars as pl
from pydantic import field_validator

from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.metrics.utils import manual_group_by


class RetrievalSuite(MetricSuite):
    """
    Retrieval metric suites.
    """

    kind: Literal["retrieval"] = "retrieval"

    @field_validator("metric_labels")
    @classmethod
    def validate_metrics_allowlist(cls, v: set[str]) -> set[str]:
        """
        Assert all metrics are in the metric_labels.
        """
        allowlist = ["retrieval_mae", "retrieval_mae_delta", "retrieval_edistance"]
        for metric in v:
            if metric not in allowlist:
                raise ValueError(f"Metric {metric} is not supported for retrieval tasks")
        return v

    def evaluate(self) -> pl.DataFrame:
        rows = []

        # Groupby context
        for context in manual_group_by(self.predictions.dataset.obs, self.predictions.context_groupby_cols):
            context_predicate = [pl.col(col) == value for col, value in context.items()]

            # Ground truth
            y_base = self.ground_truth.get_basal_states(*context_predicate)
            y_true = self.ground_truth.get_perturbed_states(*context_predicate)
            p_true = self.ground_truth.get_perturbations(*context_predicate)

            # Predictions
            y_pred = self.predictions.get_perturbed_states(*context_predicate)
            p_pred = self.predictions.get_perturbations(*context_predicate)

            for label, metric in self.metrics.items():
                if metric.is_distributional and not self.use_distributional_metrics:
                    continue
                score, supp = metric.fn(
                    y_pred=y_pred,
                    y_true=y_true,
                    y_base=y_base,
                    p_true=p_true,
                    p_pred=p_pred,
                    **metric.kwargs,
                )
                rows.append({"score": score, "metric": label, **supp, **context})

        return pl.DataFrame(rows)
