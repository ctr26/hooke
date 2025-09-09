import numpy as np
import polars as pl
import tqdm

from vcb.evaluate.utils import (
    add_compound_perturbation_to_obs,
    add_index_to_obs,
    check_disease_model_consistency,
)
from vcb.models.dataloader import BiologicalContext, Perturbation, PerturbationGroup
from vcb.models.dataset import Dataset
from vcb.models.misc import IndexSet


class DrugscreenDataloader:
    """
    A dataloader for drugscreen data.

    Takes in a dataset and a set of indices to filter the dataset by.

    This dataloader then returns triplets of paired sets of indices that describe:
      - The set of negative control states
      - The set of base states
      - The set of perturbed states

    These sets are variable in size, both across groups and within groups.
    """

    def __init__(self, dataset: Dataset, indices: IndexSet):
        self.dataset = dataset
        self.indices = indices

        self._groups: list[PerturbationGroup] = self._cache_groups()

    def _group_by_cols(self) -> list[str]:
        return self.dataset.metadata.biological_context + ["batch_center"]

    def _cache_groups(self) -> list[PerturbationGroup]:
        """
        Cache the different groups in this dataset.

        Each group can be thought of as a batch of observations that have the same biological context and same perturbations.
        """

        groups: list[PerturbationGroup] = []

        # A quick, random, sanity check on a consistent disease model before diving in
        check_disease_model_consistency(self.dataset.obs)

        # Add a column that lets us map back to the original index as we filter the observations.
        obs = add_index_to_obs(self.dataset.obs)

        # Group the observations.
        # Within each group, we'll always have the same control and base states, paired with various sets of perturbed states.
        # We need to maintain the order to ensure deterministic behavior.
        grouped = obs.group_by(self._group_by_cols(), maintain_order=True)
        total = obs[self._group_by_cols()].n_unique()

        for _, batch in tqdm.tqdm(grouped, total=total):
            # Since we group by the biological context, we know there is only one unique value for each column.
            biological_context = {
                col: batch[col].unique()[0]
                for col in self.dataset.metadata.biological_context
            }

            # Find all control indices
            control_indices = batch.filter(pl.col("is_negative_control"))[
                "original_index"
            ].to_list()

            # Find all base state indices
            base_state_indices = batch.filter(pl.col("is_base_state"))[
                "original_index"
            ].to_list()

            # Now we will want to group by unique compound perturbations
            with_compound_cols = batch.filter(
                # select only compound perturbations
                pl.col("drugscreen_query")
            ).filter(
                # only keep perturbations in this split
                pl.col("original_index").is_in(self.indices)
            )

            # We'll extract just the compound relvant info from the nested pert column, for easier grouping
            with_compound_cols = add_compound_perturbation_to_obs(with_compound_cols)

            # Aggregate indexes by unique query compounds
            for _, perturbation_groups in with_compound_cols.group_by(
                ["inchikey", "concentration"], maintain_order=True
            ):
                # Get the metadata about the perturbations.
                # Since we've grouped by perturbation, all perturbations should be the same and we can just take any one of them. (the first)
                perturbations = perturbation_groups[0, "perturbations"]

                # merge the whole sample into alist
                perturbation_indices = sorted(
                    set(perturbation_groups["original_index"].to_list())
                )
                groups.append(
                    PerturbationGroup(
                        controls=control_indices,
                        base_states=base_state_indices,
                        perturbed_states=perturbation_indices,
                        biological_context=biological_context,
                        perturbations=perturbations,
                    )
                )
        return groups

    def __len__(self):
        return len(self._groups)

    def __getitem__(
        self, index: int
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, list[Perturbation], BiologicalContext
    ]:
        group = self._groups[index]

        # Extract the features
        control_features = self.dataset.X[group.controls]
        base_features = self.dataset.X[group.base_states]
        perturbed_features = self.dataset.X[group.perturbed_states]

        # Extract the metadata
        return (
            control_features,
            base_features,
            perturbed_features,
            group.perturbations,
            group.biological_context,
        )
