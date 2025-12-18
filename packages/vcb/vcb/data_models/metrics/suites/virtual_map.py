from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from loguru import logger

from vcb.data_models.metrics.metric_info import MinimalMetricInfo
from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.metrics.map_building.efaar import map_building_pipeline
from vcb.metrics.virtual_map import (
    map_cosine_sim_classification,
    map_cosine_sim_error,
    map_cosine_sim_ranking,
)


class VirtualMapSuite(MetricSuite):
    """
    Perturbation effect prediction metric suites.
    """

    kind: Literal["virtual_map"] = "virtual_map"

    plot_destination: Path | None = None

    _all_supported_metrics: ClassVar[dict[str, MinimalMetricInfo]] = {
        "map_error": MinimalMetricInfo(fn=map_cosine_sim_error),
        "map_ranking": MinimalMetricInfo(fn=map_cosine_sim_ranking),
        "map_classification_90%": MinimalMetricInfo(fn=map_cosine_sim_classification),
        "map_classification_0.4": MinimalMetricInfo(
            fn=map_cosine_sim_classification, kwargs={"cosine_sim_threshold": 0.4}
        ),
        "map_classification_0.7": MinimalMetricInfo(
            fn=map_cosine_sim_classification, kwargs={"cosine_sim_threshold": 0.7}
        ),
    }

    def _maybe_get_subdir(self, subdir: str) -> Path | None:
        return self.plot_destination / subdir if self.plot_destination is not None else None

    def get_common_perturbations(self, true: list[str], pred: list[str]) -> list[str]:
        true_unique = set(true)
        pred_unique = set(pred)
        intersection = true_unique & pred_unique
        if true_unique != pred_unique:
            logger.warning(
                f"The true (n={len(true_unique)}) and predicted (n={len(pred_unique)}) "
                f"perturbations do not match. Using the intersection (n={len(intersection)})."
            )
        return list(intersection)

    def align_map(self, mat: np.ndarray, source_order: list[str], target_order: list[str]) -> np.ndarray:
        # Only keep the rows and columns of the shared perturbations.
        # And reorder the rows and columns to match the target order.
        indices = [source_order.index(target) for target in target_order]
        mat = mat[indices, :][:, indices]

        # Only keep the upper triangle, since the matrix is symmetric.
        mat = mat[np.triu_indices(mat.shape[0], k=1)]
        return mat

    def evaluate(self, ground_truth: TaskAdapter, predictions: TaskAdapter) -> pl.DataFrame:
        rows = []

        obs_perturbed = predictions.all_perturbed_obs
        cell_types = obs_perturbed["cell_type"].unique().to_list()
        pert_type = obs_perturbed["type"].unique()

        if len(pert_type) != 1:
            raise ValueError(f"Cannot compute maps for combined perturbation types, found: {pert_type}")
        pert_type = pert_type[0]

        if pert_type == "genetic":
            map_pert_col = "ensembl_gene_id"
        elif pert_type == "compound":
            map_pert_col = "inchikey"
        else:
            raise ValueError(f"Unknown perturbation type: {pert_type}")

        # TODO (cwognum): This probably could be cached. No need to recompute it every time.
        true_maps = {}
        logger.info("Computing the ground truth map...")
        for mapmat, cell_type, perturbations in map_building_pipeline(
            ground_truth.dataset,
            perturbation_groupby_columns=[map_pert_col],
            plot_destination=self._maybe_get_subdir("ground_truth"),
            cell_type_subset=cell_types,
        ):
            true_maps[cell_type] = (mapmat, perturbations)

        pred_maps = {}
        logger.info("Computing the predicted map...")
        for cell_type in cell_types:
            mapmat, cell_type, perturbations = next(
                map_building_pipeline(
                    predictions.dataset,
                    perturbation_groupby_columns=[map_pert_col],
                    plot_destination=self._maybe_get_subdir("predicted"),
                    cell_type_subset=[cell_type],
                    perturbation_order=true_maps[cell_type][1],
                )
            )
            pred_maps[cell_type] = (mapmat, perturbations)

        for cell_type, (true_map, true_perturbations) in true_maps.items():
            pred_map, pred_perturbations = pred_maps[cell_type]

            common = self.get_common_perturbations(true_perturbations, pred_perturbations)
            if len(common) == 0:
                logger.warning(f"No common perturbations found for cell type {cell_type}. Skipping.")
                continue
            if len(common) < len(true_perturbations) or len(common) < len(pred_perturbations):
                logger.warning(
                    f"Predictions and ground truth don't have the same perturbations for cell type {cell_type}."
                )

            y_true = self.align_map(true_map, true_perturbations, common)
            y_pred = self.align_map(pred_map, pred_perturbations, common)

            # Compute performance measures
            for label, metric in self.metrics.items():
                scores = metric.fn(y_true, y_pred, **metric.kwargs)
                for k, v in scores.items():
                    rows.append({"metric": label + "_" + k, "score": v, "cell_type": cell_type})

        return pl.DataFrame(rows)
