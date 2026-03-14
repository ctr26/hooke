from typing import Literal, Self, Tuple

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from vcb.data_models.task.base import TaskAdapter
from vcb.utils import filter_negative_controls_and_duplicates


def add_single_perturbation_to_obs(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Add the single perturbation to the observations.
    """
    # first skip the negative controls and duplicates
    obs = filter_negative_controls_and_duplicates(obs)

    # assert single perts only (or empty)
    singles_and_empties = obs.filter(pl.col("clean_perturbations").list.len() <= 1)
    if not singles_and_empties.shape[0] == obs.shape[0]:
        raise ValueError(
            f"Expected all observations to be singles, but singles comprised "
            f"{singles_and_empties.shape[0]} / {obs.shape[0]} of the observations."
        )
    return (
        obs.with_columns(pl.col("clean_perturbations").alias("pert_ex_un"))
        .explode("pert_ex_un")
        .unnest("pert_ex_un")
    )


class SinglesTaskAdapter(TaskAdapter):
    """
    A dataset for single perturbations.

    Args:
        perturbation_groupby_cols_types: The columns to group by for the perturbations.
        context_groupby_cols: The columns to group by for the context.
        perturbation_length_filter: The filter for acceptable perturbations column lengths; so as to
           keep valid perturbations and controls for the target task. This can be very dataset specific,
           even for singles. For instance a PerturbSeq dataset with an exclusive mock negative of non-targeting will be [1],
           whereas datasets where the active is applied on top, and not in exclusion to the mock could be [0, 1] or [1, 2],
           depending on whether the mock is labeled explicitly in the data or just always there such as DMSO in RXRX data.

    """

    kind: Literal["singles"] = "singles"

    batch_groupby_cols: list[str] = Field(default_factory=lambda: ["batch_center"])
    perturbation_groupby_cols_types: list[Tuple[str, str]] = Field(
        default_factory=lambda: [("inchikey", "<U27"), ("concentration", "float")]
    )
    context_groupby_cols: list[str] = Field(default_factory=lambda: [])
    perturbation_splitting_col: str = Field(default="inchikey")

    @model_validator(mode="before")
    def set_from_metadata(self) -> Self:
        """
        Set context_groupby_cols and perturbation_splitting_col from dataset metadata if available.
        """
        if self["dataset"].metadata is not None:
            self["context_groupby_cols"] = list(self["dataset"].metadata.biological_context)

        return self

    @property
    def perturbation_length_filter(self) -> pl.Expr:
        return pl.col("perturbations").list.len().le(1)

    def get_all_perturbed_obs(self) -> pl.DataFrame:
        # define as, 'has a query', i.e. contains at least one perturbation with usage_class == "query"
        return self.dataset.obs.filter(
            pl.col("perturbations").list.eval(pl.element().struct.field("usage_class").eq("query")).list.any()
        )

    def get_all_basal_obs(self) -> pl.DataFrame:
        return self.dataset.obs.filter(pl.col("is_negative_control"))

    def prepare(self) -> None:
        """
        Any (costly) operations that need to be done once on the dataset should be done here.
        """
        if self.dataset._obs_is_prepared:
            # temp solution to avoid changes to mutable dataset that can overwrite each other
            # or otherwise lead to cryptic behavior
            raise ValueError("can't call prepare on a prepared dataset and be confident in results")

        # Since multiple adapter instances can reference the same dataset,
        # some of the preprocessing here may already have been done, even if _is_prepared is False.
        # Out of precaution, we reset and recompute.

        obs = self.dataset.obs
        obs = obs.with_row_index("original_index")
        obs = add_single_perturbation_to_obs(obs)
        self.dataset.obs = obs

        self._all_basal_obs_cache = None
        self._all_perturbed_obs_cache = None
        self.dataset._obs_is_prepared = True

    def get_basal_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.all_basal_obs.filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]

    def get_perturbations(self, *predictates: pl.Expr) -> np.ndarray:
        # Filter
        obs = self.all_perturbed_obs.filter(*predictates)

        # Get the perturbations and convert to numpy array
        perturbations = list(obs[sorted(self.perturbation_groupby_cols)].iter_rows())
        dt = np.dtype(sorted(self.perturbation_groupby_dtype))
        perturbations = np.array(perturbations, dtype=dt)

        return perturbations

    def get_perturbed_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.all_perturbed_obs.filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]
