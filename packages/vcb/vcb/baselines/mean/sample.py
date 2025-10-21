from collections import defaultdict
from typing import Annotated, Any, Self, TypeAlias

import numpy as np
from loguru import logger
from pydantic import PrivateAttr

from vcb.baselines.mean.base import BaseMeanBaseline
from vcb.baselines.utils import dict_to_ordered_tuple
from vcb.data_models.task.base import TaskAdapter

_MAPPING_TYPE: TypeAlias = Annotated[
    dict[tuple, dict[tuple, list[tuple]]],
    PrivateAttr(default_factory=lambda: defaultdict(lambda: defaultdict(list))),
]


def sample_random_key(mapping: dict[tuple, Any] | list[tuple]) -> tuple:
    """Since the keys are tuples, we can't use np.random.choice directly."""
    if isinstance(mapping, dict):
        return sample_random_key(list(mapping.keys()))
    else:
        rng_index = np.random.randint(0, len(mapping))
        return mapping[rng_index]


class MeanContextSampleBaseline(BaseMeanBaseline):
    """
    Mean context sample baseline.

    Main idea: Return a random perturbed state for the same context.

    Protocol:
        (1) Sample a random perturbation for the given context.
        (2) Sample a random batch for that context-perturbation pair.
        (3) Sample a random perturbed state from that batch. This is our prediction.
    """

    _mapping: _MAPPING_TYPE

    def fit(self, task: TaskAdapter) -> Self:
        super().fit(task)
        for context, batch, perturbation in self.mean_perturbed_states.keys():
            self._mapping[context][perturbation].append(batch)
        return self

    def predict(self, task: TaskAdapter) -> np.ndarray:
        predictions = []
        missing_count = 0
        for row in task.dataset.obs.iter_rows(named=True):
            context = dict_to_ordered_tuple(row, task.context_groupby_cols)

            if context not in self._mapping:
                missing_count += 1
                context = sample_random_key(self._mapping)

            # (1) Sample a random perturbation
            perturbation = sample_random_key(self._mapping[context])

            # (2) Sample a random batch
            batch = sample_random_key(self._mapping[context][perturbation])

            # (3) Make the prediction
            prediction = self.mean_perturbed_states[context, batch, perturbation]
            predictions.append(prediction)

        if missing_count > 0:
            missing_percentage = missing_count / len(task.dataset.obs) * 100
            logger.warning(
                f"For {missing_count} ({missing_percentage:.2f}%) contexts, "
                "we did not find the context in the training set. "
                "In these cases we simply return a random perturbed state."
            )

        return np.vstack(predictions)


class MeanPerturbationSampleBaseline(BaseMeanBaseline):
    """
    Mean perturbation sample baseline.

    Main idea: Return a random perturbed state for the same perturbation.

    Protocol:
        (1) Sample a random context for the given perturbation.
        (2) Sample a random batch from that context.
        (3) Sample a random perturbed state from that batch. This is our prediction.
    """

    _mapping: _MAPPING_TYPE

    def fit(self, task: TaskAdapter) -> Self:
        super().fit(task)
        for context, batch, perturbation in self.mean_perturbed_states.keys():
            self._mapping[perturbation][context].append(batch)
        return self

    def predict(self, task: TaskAdapter) -> np.ndarray:
        predictions = []
        missing_count = 0

        for row in task.dataset.obs.iter_rows(named=True):
            perturbation = dict_to_ordered_tuple(row, task.perturbation_groupby_cols)

            if perturbation not in self._mapping:
                missing_count += 1
                perturbation = sample_random_key(self._mapping)

            # (1) Sample a random perturbation
            context = sample_random_key(self._mapping[perturbation])

            # (2) Sample a random batch
            batch = sample_random_key(self._mapping[perturbation][context])

            # (3) Make the prediction
            prediction = self.mean_perturbed_states[context, batch, perturbation]
            predictions.append(prediction)

        if missing_count > 0:
            missing_percentage = missing_count / len(task.dataset.obs) * 100
            logger.warning(
                f"For {missing_count} ({missing_percentage:.2f}%) perturbations, "
                "we did not find the perturbation in the training set. "
                "In these cases we simply return a random perturbed state."
            )

        return np.vstack(predictions)
