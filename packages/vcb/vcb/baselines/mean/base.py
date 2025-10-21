from collections import defaultdict
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field

from vcb.baselines.base import BaseBaseline
from vcb.baselines.utils import dict_to_ordered_tuple
from vcb.data_models.task.base import TaskAdapter
from vcb.utils import predicate_group_by


def _mean_or_squeeze(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0) if X.shape[0] > 1 else X.squeeze()


class BaseMeanBaseline(BaseBaseline):
    """
    Any baseline that relies on means computed
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean_perturbed_states: dict[tuple, np.ndarray] = Field(default_factory=dict)
    mean_basal_states: dict[tuple, np.ndarray] = Field(default_factory=dict)
    mean_deltas: dict[tuple, dict[tuple, np.ndarray]] | None = None

    def cache_means(self, task: TaskAdapter) -> None:
        """
        Cache the means of the context by groups.

        We compute and cache three different means here:
          1. The mean of the basal states per context such that each batch is weighted equally.
          2. The mean of the perturbed states per context-batch-perturbation triplet.
          3. The mean delta, which is the difference between the mean perturbed state of (2)
             and the corresponding basal state of (1), aggregated across batches.
        """

        for context, context_obs, context_predicate in predicate_group_by(
            task.dataset.obs, task.context_groupby_cols, description="Grouping by context"
        ):
            context = dict_to_ordered_tuple(context, task.context_groupby_cols)
            mean_base_states = []
            for batch, batch_obs, batch_predicate in predicate_group_by(
                context_obs, task.batch_groupby_cols, description="Grouping by batch"
            ):
                batch = dict_to_ordered_tuple(batch, task.batch_groupby_cols)

                # Get the basal states for this context-batch pair
                basal_predicate = context_predicate + batch_predicate
                X_basal = task.get_basal_states(*basal_predicate)
                mean_base_states.append(_mean_or_squeeze(X_basal))

                for perturbation, _, perturbation_predicate in predicate_group_by(
                    batch_obs, task.perturbation_groupby_cols, description="Grouping by perturbation"
                ):
                    # Get the perturbed states for this context-batch-perturbation triplet
                    perturbed_predicate = basal_predicate + perturbation_predicate
                    X_perturbed = task.get_perturbed_states(*perturbed_predicate)

                    # (2)
                    perturbation = dict_to_ordered_tuple(perturbation, task.perturbation_groupby_cols)
                    self.mean_perturbed_states[context, batch, perturbation] = _mean_or_squeeze(X_perturbed)

            # (1)
            self.mean_basal_states[context] = _mean_or_squeeze(np.vstack(mean_base_states))

        # Compute the mean delta for each context-perturbation pair, aggregated across batches
        mean_deltas = defaultdict(lambda: defaultdict(list))
        for (context, _, perturbation), X_perturbed in self.mean_perturbed_states.items():
            X_basal = self.mean_basal_states[context]
            delta = X_perturbed - X_basal
            mean_deltas[context][perturbation].append(delta)

        self.mean_deltas = defaultdict(dict)
        for context, deltas_per_context in mean_deltas.items():
            for perturbation, deltas_per_perturbation in deltas_per_context.items():
                deltas = np.vstack(deltas_per_perturbation)
                # (3)
                self.mean_deltas[context][perturbation] = _mean_or_squeeze(deltas)

    def fit(self, task: TaskAdapter) -> Self:
        self.cache_means(task)
        return self
