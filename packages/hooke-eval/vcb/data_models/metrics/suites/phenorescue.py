from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from loguru import logger
from pydantic import Field, computed_field
from tqdm import tqdm

from vcb.data_models.metrics.metric_info import MinimalMetricInfo
from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.metrics.drugscreen.rescue_screen import rescue_screen_analysis
from vcb.metrics.phenorescue import hit_classification, hit_ranking, hit_score_error
from vcb.settings import settings


class PhenorescueSuite(MetricSuite):
    """
    Perturbation effect prediction metric suites.
    """

    kind: Literal["phenorescue"] = "phenorescue"

    embedding: Literal["txam", "pca"] | None = None
    embedding_kwargs: dict = Field(default_factory=dict)

    plot_hit_threshold: float | None = 0.5

    random_state: int = 42

    _all_supported_metrics: ClassVar[dict[str, MinimalMetricInfo]] = {
        "hit_score_error": MinimalMetricInfo(fn=hit_score_error),
        "hit_ranking": MinimalMetricInfo(fn=hit_ranking),
        "hit_classification": MinimalMetricInfo(fn=hit_classification),
    }

    @computed_field
    def save_dir(self) -> Path:
        return settings.ensure_save_dir(self.kind)

    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        rows = []

        # No need to also check the predcitions, since the EvaluationConfig already asserts
        # that the ground truth and predictions are of the same type.
        if not isinstance(ground_truth, DrugscreenTaskAdapter):
            raise ValueError(f"The {self.kind} suite can only be evaluated on drugscreen data")

        # Get the hit scores for the ground truth
        # TODO (cwognum): Cache this. No need to recompute it every time.
        y_true = {}
        plot_compounds = {}
        logger.info("Computing the hit scores for the ground truth...")
        for experiment, hit_scores, _, exp_plot_compounds in rescue_screen_analysis(
            ground_truth.dataset,
            plot_destination=self.save_dir / "plots" / "ground_truth",
            plot_hit_threshold=self.plot_hit_threshold,
            embedding=self.embedding,
            embedding_kwargs=self.embedding_kwargs,
            random_state=self.random_state,
        ):
            y_true[experiment] = hit_scores
            plot_compounds[experiment] = exp_plot_compounds

        # Get the hit scores for the predictions
        # NOTE (cwognum): For this to work, the predictions need to specify the negative controls and base states as well.
        #   We could simply copy these from the ground truth, but it's an open question whether it would be more performant to predict these too.
        #   We should thus provide the flexibility to specify the negative controls and base states in the predictions.
        y_pred = {}
        logger.info("Computing the hit scores for the predictions...")
        for experiment, hit_scores, _, _ in rescue_screen_analysis(
            predictions.dataset,
            plot_destination=self.save_dir / "plots" / "predicted",
            plot_compounds=plot_compounds,
            embedding=self.embedding,
            embedding_kwargs=self.embedding_kwargs,
            random_state=self.random_state,
        ):
            y_pred[experiment] = hit_scores

        # Evaluate the metrics
        for experiment in tqdm(
            y_true.keys(), desc="Evaluating accuracy of the predicted hit scores per experiment", leave=False
        ):
            y_true_experiment = y_true[experiment]
            y_pred_experiment = y_pred[experiment]

            # Consistently order hit scores
            inchikeys = sorted(y_true_experiment.keys())
            y_true_experiment = np.array([y_true_experiment[k] for k in inchikeys])
            y_pred_experiment = np.array([y_pred_experiment[k] for k in inchikeys])

            # Compute performance measures
            for label, metric in self.metrics.items():
                scores = metric.fn(y_true_experiment, y_pred_experiment, **metric.kwargs)
                for k, v in scores.items():
                    rows.append({"metric": label + "_" + k, "score": v, "experiment": experiment})

        return pl.DataFrame(rows)
