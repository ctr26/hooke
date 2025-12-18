from collections import defaultdict
from typing import Self

import numpy as np
from loguru import logger
from pydantic import PrivateAttr

from vcb.baselines.mean.base import BaseMeanBaseline
from vcb.baselines.utils import dict_to_ordered_tuple
from vcb.data_models.task.base import TaskAdapter


class MeanContextDeltaBaseline(BaseMeanBaseline):
    """
    Mean context delta baseline.

    Main idea: What if we assume that all perturbations have the same effect on a given context (e.g. cell type or disease model)?

    Protocol:
        (1) Get the mean base state from the same context, aggregated across batches.
        (2) Get the mean delta for the same context, aggregated across batches and perturbations.
        (3) The prediction is (1) + (2).

    Grouping by context in (2), rather than perturbation, is what differentiates this baseline.
    """

    _mean_deltas_per_context: dict[tuple, np.ndarray] = PrivateAttr(default_factory=dict)

    def fit(self, task: TaskAdapter) -> Self:
        super().fit(task)

        # For this baseline, we compute the delta _per context_.
        # These deltas are already aggregated across batches.
        # We thus need to aggregate the deltas across perturbations, weighing each perturbation equally.

        for context, deltas_per_context in self.mean_deltas.items():
            deltas = []
            for _, delta in deltas_per_context.items():
                deltas.append(delta)
            deltas = np.vstack(deltas)

            self._mean_deltas_per_context[context] = np.mean(deltas, axis=0)
        return self

    def predict(self, task: TaskAdapter) -> np.ndarray:
        predictions = []
        missing_count = 0

        # Loop over the entire dataset.
        for row in task.dataset.obs.iter_rows(named=True):
            # Get the "keys" we need to index the cached means
            context = dict_to_ordered_tuple(row, task.context_groupby_cols)

            # (1)
            # NOTE (cwognum): Don't remove this .copy(). You'll keep updating the mean basal state in place.
            # Trust me, I've spent a Friday evening slowly going insane trying to figure out what's happening.
            prediction = self.mean_basal_states[context].copy()

            # (2) and (3)
            if context in self._mean_deltas_per_context:
                prediction += self._mean_deltas_per_context[context]
            else:
                missing_count += 1

            # Clip to non-negative values.
            prediction = np.clip(prediction, 0, None)
            predictions.append(prediction)

        if missing_count > 0:
            missing_percentage = missing_count / len(task.dataset.obs) * 100
            logger.warning(
                f"For {missing_count} ({missing_percentage:.2f}%) contexts, "
                "we did not find the context in the training set. "
                "In these cases we simply return the basal state (i.e. no delta)."
            )

        return np.vstack(predictions)


class MeanPerturbationDeltaBaseline(BaseMeanBaseline):
    """
    Mean perturbation delta baseline.

    Main idea: What if we assume that a specific perturbation has the same effect in every context (e.g. cell type or disease model)?

    Protocol:
        (1) Get the mean base state from the same context, aggregated across batches.
        (2) Get the mean delta for the same perturbation, aggregated across batches and contexts.
            If the perturbation is not found, we simply return the mean basal state.
        (3) The prediction is (1) + (2).

    Grouping by perturbation in (2), rather than context, is what differentiates this baseline.
    """

    _mean_deltas_per_perturbation: dict[tuple, np.ndarray] = PrivateAttr(default_factory=dict)

    def fit(self, task: TaskAdapter) -> Self:
        super().fit(task)

        # For this baseline, we compute the delta _per perturbation_.
        deltas = defaultdict(list)
        for _, deltas_per_context in self.mean_deltas.items():
            for perturbation, delta in deltas_per_context.items():
                deltas[perturbation].append(delta)

        for perturbation, deltas in deltas.items():
            self._mean_deltas_per_perturbation[perturbation] = np.mean(np.vstack(deltas), axis=0)
        return self

    def predict(self, task: TaskAdapter) -> np.ndarray:
        predictions = []
        missing_count = 0

        # Loop over the entire dataset.
        for row in task.dataset.obs.iter_rows(named=True):
            # Get the "keys" we need to index the cached means
            perturbation = dict_to_ordered_tuple(row, task.perturbation_groupby_cols)
            context = dict_to_ordered_tuple(row, task.context_groupby_cols)

            # (1)
            prediction = self.mean_basal_states[context].copy()

            # (2) and (3)
            if perturbation in self._mean_deltas_per_perturbation:
                prediction += self._mean_deltas_per_perturbation[perturbation]
            else:
                missing_count += 1

            # Clip to non-negative values.
            prediction = np.clip(prediction, 0, None)
            predictions.append(prediction)

        if missing_count > 0:
            missing_percentage = missing_count / len(task.dataset.obs) * 100
            logger.warning(
                f"For {missing_count} ({missing_percentage:.2f}%) perturbations, "
                "we did not find the perturbation in the training set. "
                "In these cases we simply return the basal state (i.e. no delta)."
            )

        return np.vstack(predictions)
