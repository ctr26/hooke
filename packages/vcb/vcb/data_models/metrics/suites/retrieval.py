from typing import Literal

import polars as pl
from pydantic import field_validator

from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.utils import predicate_group_by


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

    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        rows = []

        # Groupby context
        for group_label, _, predicate in predicate_group_by(
            predictions.dataset.obs,
            predictions.context_groupby_cols,
            description=f"Computing {self.kind} suite per {predictions.context_groupby_cols}",
        ):
            # Ground truth
            y_base = ground_truth.get_basal_states(*predicate)
            y_true = ground_truth.get_perturbed_states(*predicate)
            p_true = ground_truth.get_perturbations(*predicate)

            # Predictions
            y_pred = predictions.get_perturbed_states(*predicate)
            p_pred = predictions.get_perturbations(*predicate)

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
                rows.append({"score": score, "metric": label, **supp, **group_label})

        return pl.DataFrame(rows)
