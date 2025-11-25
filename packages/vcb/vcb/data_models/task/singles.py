import numpy as np
import polars as pl
from pydantic import Field
from typing import Literal, Tuple

from vcb.data_models.task.base import TaskAdapter


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
        default_factory=lambda: [("ensembl_gene_id", "<U27")]
    )
    context_groupby_cols: list[str] = Field(default_factory=lambda: ["cell_line"])
    perturbation_splitting_col: str = Field(default="ensembl_gene_id")

    _filtered_perturbed_obs: pl.DataFrame | None = None
    _filtered_basal_obs: pl.DataFrame | None = None

    @property
    def perturbation_length_filter(self) -> pl.Expr:
        # note that this acts _after_ prepare has been called
        # which in this subclass already asserts that the perturbations are length = 1
        return pl.lit(True)

    def get_all_perturbed_obs(self) -> pl.DataFrame:
        if self._filtered_perturbed_obs is None:
            self._filtered_perturbed_obs = self.dataset.obs.filter(~pl.col("is_negative_control"))

        return self._filtered_perturbed_obs

    def get_all_basal_obs(self) -> pl.DataFrame:
        if self._filtered_basal_obs is None:
            self._filtered_basal_obs = self.dataset.obs.filter(pl.col("is_negative_control"))

        return self._filtered_basal_obs

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
        # assert single perts only
        if not obs.filter(pl.col("perturbations").list.len().eq(1)).shape[0] == obs.shape[0]:
            raise ValueError("Only single genetic perturbations are supported")
        obs = (
            obs.with_columns(pl.col("perturbations").alias("pert_ex_un"))
            .explode("pert_ex_un")
            .unnest("pert_ex_un")
        )
        self.dataset.obs = obs

        self.dataset._obs_is_prepared = True
        self._filtered_perturbed_obs = None
        self._filtered_basal_obs = None

    def get_basal_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.get_all_basal_obs().filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]

    def get_perturbations(self, *predictates: pl.Expr) -> np.ndarray:
        # Filter
        obs = self.get_all_perturbed_obs().filter(*predictates)

        # Get the perturbations and convert to numpy array
        perturbations = list(obs[sorted(self.perturbation_groupby_cols)].iter_rows())
        dt = np.dtype(sorted(self.perturbation_groupby_dtype))
        perturbations = np.array(perturbations, dtype=dt)

        return perturbations

    def get_perturbed_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.get_all_perturbed_obs().filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]
