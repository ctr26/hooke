from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import polars as pl
from loguru import logger
from pydantic import computed_field

from vcb.data_models.metrics.metric_info import MinimalMetricInfo
from vcb.data_models.metrics.suite import MetricSuite
from vcb.data_models.task.base import TaskAdapter
from vcb.metrics.map_building.efaar import map_building_pipeline
from vcb.metrics.virtual_map import (
    map_cosine_sim_classification,
    map_cosine_sim_error,
    map_cosine_sim_ranking,
)
from vcb.settings import settings


class VirtualMapSuite(MetricSuite):
    """
    Perturbation effect prediction metric suites.
    """

    kind: Literal["virtual_map"] = "virtual_map"

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

    @computed_field
    def save_dir(self) -> Path:
        return settings.ensure_save_dir(self.kind)

    @computed_field
    def cache_dir(self) -> Path:
        return settings.cache_dir / self.kind

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

        true_maps = {}
        logger.info("Computing the ground truth map...")
        for vmap in map_building_pipeline(
            ground_truth.dataset,
            perturbation_groupby_columns=[map_pert_col],
            save_destination=self.save_dir / "ground_truth",
            cell_type_subset=cell_types,
            cache_dir=self.cache_dir / ground_truth.dataset.dataset_id / "ground_truth",
        ):
            true_maps[vmap.cell_type] = vmap

        pred_maps = {}
        logger.info("Computing the predicted map...")
        for cell_type in cell_types:
            pred_maps[cell_type] = next(
                map_building_pipeline(
                    predictions.dataset,
                    perturbation_groupby_columns=[map_pert_col],
                    save_destination=self.save_dir / "predicted",
                    cell_type_subset=[cell_type],
                    perturbation_order=true_maps[cell_type].perturbations,
                )
            )

        for cell_type, true_map in true_maps.items():
            pred_map = pred_maps[cell_type]

            common = self.get_common_perturbations(true_map.perturbations, pred_map.perturbations)
            if len(common) == 0:
                logger.warning(f"No common perturbations found for cell type {cell_type}. Skipping.")
                continue
            if len(common) < len(true_map.perturbations) or len(common) < len(pred_map.perturbations):
                logger.warning(
                    f"Predictions and ground truth don't have the same perturbations for cell type {cell_type}."
                )

            y_true = self.align_map(true_map.similarity_matrix, true_map.perturbations, common)
            y_pred = self.align_map(pred_map.similarity_matrix, pred_map.perturbations, common)

            # Compute performance measures
            for label, metric in self.metrics.items():
                scores = metric.fn(y_true, y_pred, **metric.kwargs)
                for k, v in scores.items():
                    rows.append({"metric": label + "_" + k, "score": v, "cell_type": cell_type})

        return pl.DataFrame(rows)
