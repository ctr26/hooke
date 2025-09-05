import numpy as np
import polars as pl
import tqdm

from vcb.models.dataloader import BiologicalContext, Perturbation, PerturbationGroup
from vcb.models.dataset import Dataset
from vcb.models.misc import IndexSet


def from_perturbations_to_disease_model(perturbations: list[dict]) -> str:
    """
    Given a list of perturbations, return the disease model.
    For drugscreen data, we can assume that it's the first perturbation in the list.

    If there is no perturbations or the first perturbation is not a genetic perturbation, return None.
    This can happen for positive controls or empties, for example.
    """

    if len(perturbations) == 0:
        return None

    sorted_perturbations = sorted(
        perturbations, key=lambda x: x["hours_post_reference"]
    )

    # Should be fine, but a quick sanity check won't hurt.
    first_perturbation = sorted_perturbations[0]
    if first_perturbation["type"] != "genetic":
        return None

    return first_perturbation["ensembl_gene_id"]


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

        # Add a column that lets us map back to the original index as we filter the observations.
        obs = self.dataset.obs.with_columns(
            pl.Series(name="original_index", values=range(len(self.dataset.obs)))
        )

        # Add a column with the disease model:
        obs = obs.with_columns(
            pl.col("perturbations")
            .map_elements(from_perturbations_to_disease_model, return_dtype=pl.Utf8)
            .alias("disease_model")
        )

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

            to_group = batch.filter(pl.col("disease_model").is_not_null())
            for _, group in to_group.group_by("disease_model", maintain_order=True):
                # Find all base state indices
                base_state_indices = group.filter(pl.col("is_base_state"))[
                    "original_index"
                ].to_list()

                # Now group by unique perturbations
                # Flatten the perturbation into a table, making sure to keep the original index
                flat_perturbations = group.explode("perturbations").unnest(
                    "perturbations"
                )["original_index", "drugscreen_query", "inchikey", "concentration"]

                # (1) Only keep the perturbations in this split
                filtered_perturbations = flat_perturbations.filter(
                    pl.col("original_index").is_in(self.indices)
                )

                # (2) Only keep the actual perturbations
                filtered_perturbations = filtered_perturbations.filter(
                    pl.col("drugscreen_query")
                )

                # (3) Only keep the perturbatins with a valid ID
                filtered_perturbations = filtered_perturbations.filter(
                    ~pl.col("inchikey").is_null()
                )

                # Aggregate indexes by unique query compounds
                for perturbation_indices in (
                    filtered_perturbations.group_by(
                        ["inchikey", "concentration"], maintain_order=True
                    )
                    .agg(pl.col("original_index").alias("original_index"))
                    .get_column("original_index")
                    .to_list()
                ):
                    # Get the metadata about the perturbations.
                    # Since we've grouped by perturbation, all perturbations should be the same and we can just take any one of them.
                    perturbations = self.dataset.obs[
                        perturbation_indices[0], "perturbations"
                    ].to_list()

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
