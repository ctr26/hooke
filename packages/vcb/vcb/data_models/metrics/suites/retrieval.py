from typing import ClassVar, Literal

import polars as pl

from vcb.data_models.metrics.metric_info import MinimalMetricInfo
from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.metrics.retrieval import calculate_edistance_retrieval, calculate_mae_retrieval
from vcb.utils import predicate_group_by


class RetrievalMetricInfo(MinimalMetricInfo):
    """
    Retrieval metric metadata.
    """

    is_distributional: bool = False


class RetrievalSuite(MetricSuite):
    """
    Retrieval metric suites.
    """

    kind: Literal["retrieval"] = "retrieval"

    use_distributional_metrics: bool = True

    _all_supported_metrics: ClassVar[dict[str, RetrievalMetricInfo]] = {
        "retrieval_mae": RetrievalMetricInfo(fn=calculate_mae_retrieval),
        "retrieval_edistance": RetrievalMetricInfo(fn=calculate_edistance_retrieval, is_distributional=True),
    }

    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        rows = []

        # Groupby context
        for group_label, _, predicate in predicate_group_by(
            predictions.get_all_perturbed_obs(),
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
