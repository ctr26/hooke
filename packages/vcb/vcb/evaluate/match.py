from typing import Generator

import numpy as np
import polars as pl

from vcb.evaluate.utils import (
    add_compound_perturbation_to_obs,
    check_disease_model_consistency,
)
from vcb.models.dataset import Dataset
from vcb.models.predictions import Predictions


def yield_batch_paired_dataframes(
    predictions: Predictions, ground_truth: Dataset
) -> Generator[tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]]:
    """
    Match predictions to ground truth.
    """
    obs_pred = predictions.obs
    obs_truth = ground_truth.obs

    # Check to ensure that we don't need to group by disease model
    check_disease_model_consistency(obs_truth)
    check_disease_model_consistency(obs_pred)

    # An index to maintain a reference to the observation in the original dataframe.
    obs_pred = obs_pred.with_row_index("original_index")
    obs_truth = obs_truth.with_row_index("original_index")

    # Predictions will likely only have perturbed state observations (i.e. no base states or controls).
    # I nevertheless expect the flag to exist, for a more robust matching.
    obs_pred = obs_pred.filter(pl.col("drugscreen_query"))
    obs_pred = add_compound_perturbation_to_obs(obs_pred)

    group_by = ["batch_center"] + ground_truth.metadata.biological_context

    # Yield the paired predictions and ground truth for each group.
    for (batch_center, *biological_context), prediction_group in obs_pred.group_by(
        group_by
    ):
        # Context that describes the specific batch we're returning here.
        context = dict(
            zip(ground_truth.metadata.biological_context, biological_context)
        )

        # Filter the ground truth to the same group.
        ground_truth_group = obs_truth.filter(pl.col("batch_center").eq(batch_center))
        for k, v in context.items():
            ground_truth_group = ground_truth_group.filter(pl.col(k).eq(v))

        yield (
            prediction_group,
            ground_truth_group,
            {
                "batch_center": batch_center,
                **context,
            },
        )


def yield_batch_pairs(
    predictions: Predictions, ground_truth: Dataset
) -> Generator[
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str], list[str], list[str]]
]:
    """
    Match predictions to ground truth.

    We match on biological context and batch center.
    Within the context of the drugscreen dataset, we assume that within a batch (i.e. plate) there is only one disease model.

    Retrieval metrics are currently computed on a batch level, and then aggregated up to the test set level.
    """
    for y_pred, y_truth, batch_definition in yield_batch_paired_dataframes(
        predictions, ground_truth
    ):
        y_pred_indices = y_pred["original_index"].to_list()

        perturbed = y_truth.filter(pl.col("drugscreen_query"))
        perturbed = add_compound_perturbation_to_obs(
            perturbed, assume_only_two_perturbations=False
        )

        base = y_truth.filter(pl.col("is_base_state"))

        y_truth_perturbed_indices = perturbed["original_index"].to_list()
        y_truth_base_indices = base["original_index"].to_list()

        X_pred = predictions.X[y_pred_indices]
        X_truth = ground_truth.X[y_truth_perturbed_indices]
        X_base = ground_truth.X[y_truth_base_indices]

        yield (
            X_pred,
            X_truth,
            X_base,
            batch_definition,
            list(y_pred["inchikey", "concentration"].iter_rows()),
            list(perturbed["inchikey", "concentration"].iter_rows()),
        )


def yield_compound_pairs(
    predictions: Predictions, ground_truth: Dataset
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]]:
    """
    Match predictions to ground truth.

    We match on perturbation (compound + concentration), biological context, and batch center.
    Within the context of the drugscreen dataset, we assume that within a batch (i.e. plate) there is only one disease model.

    Most metrics are computed on a per-compound basis, and then aggregated up to the test set level.
    """

    for y_pred, y_truth, batch_definition in yield_batch_paired_dataframes(
        predictions, ground_truth
    ):
        # Base states are the same across all perturbations.
        base_states = y_truth.filter(pl.col("is_base_state"))

        for (inchikey, concentration), prediction_group in y_pred.group_by(
            ["inchikey", "concentration"]
        ):
            pert_states = add_compound_perturbation_to_obs(
                y_truth.filter(pl.col("drugscreen_query")),
                assume_only_two_perturbations=False,
            )
            pert_states = pert_states.filter(pl.col("inchikey").eq(inchikey))
            pert_states = pert_states.filter(pl.col("concentration").eq(concentration))

            # Yield the paired predictions and ground truth for this group.
            X_base_states = ground_truth.X[base_states["original_index"].to_list()]
            X_ground_truth = ground_truth.X[pert_states["original_index"].to_list()]
            X_predicted = predictions.X[prediction_group["original_index"].to_list()]

            yield (
                X_predicted,
                X_ground_truth,
                X_base_states,
                {
                    "inchikey": inchikey,
                    "concentration": concentration,
                    **batch_definition,
                },
            )
