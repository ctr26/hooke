from typing import Literal

import numpy as np
import polars as pl
from loguru import logger
from pydantic import Field
from typing import Tuple

from vcb.data_models.task.base import TaskAdapter


def from_perturbations_to_disease_model(perturbations: list[dict]) -> str:
    """
    Given a list of perturbations, return the disease model.
    For drugscreen data, we can assume that it's the first perturbation in the list.

    If there is no perturbations or the first perturbation is not a genetic perturbation, return None.
    This can happen for positive controls or empties, for example.
    """

    if len(perturbations) == 0:
        return None

    sorted_perturbations = sorted(perturbations, key=lambda x: x["hours_post_reference"])

    # Should be fine, but a quick sanity check won't hurt.
    first_perturbation = sorted_perturbations[0]
    if first_perturbation["type"] != "genetic":
        return None

    return first_perturbation["ensembl_gene_id"]


def check_disease_model_consistency(obs: pl.DataFrame) -> None:
    """
    Within the context of the drugscreen dataset, we assume that within a batch (i.e. plate) there is only one disease model.
    This function will raise an error if this is not the case.
    """
    disease_obs = obs.filter(pl.col("is_base_state") | pl.col("drugscreen_query"))

    for i in np.random.randint(0, disease_obs.shape[0], size=5):
        i = int(i)

        # Get the disease model from the perturbations
        perturbations = disease_obs[i, "perturbations"]
        found = from_perturbations_to_disease_model(perturbations)

        # Get the disease model from the preprocessed metadata
        expected = disease_obs[i, "plate_disease_model"]

        assert found == expected, (
            f"Re-queried disease model: {found} != expected: {expected} in {disease_obs[i, 'experiment_label']}; is this standardized drugscreen data?"
        )


def add_compound_perturbation_to_obs(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Add the compound perturbation to the observations, including the inchikey and concentration.
    """

    # Split into drugscreen queries and non-drugscreen queries
    mask = pl.col("drugscreen_query") & pl.col("perturbations").list.len().eq(2)
    drugscreen_obs = obs.filter(mask)
    non_drugscreen_obs = obs.filter(~mask)

    # Process drugscreen queries with existing logic
    if len(drugscreen_obs) > 0:
        # explode = flatten list of perturbation
        # unnest = turn dict/struct into columns
        drugscreen_col = (
            drugscreen_obs.explode("perturbations")
            .unnest("perturbations")
            .filter(pl.col("inchikey").is_not_null())
            .select("inchikey", "concentration")
        )

        drugscreen_obs = drugscreen_obs.with_columns(drugscreen_col)

    # For non-drugscreen queries, set inchikey and concentration to None
    if len(non_drugscreen_obs) > 0:
        non_drugscreen_obs = non_drugscreen_obs.with_columns(
            [pl.lit(None).alias("inchikey"), pl.lit(None).alias("concentration")]
        )

    # Combine the results
    if len(drugscreen_obs) > 0 and len(non_drugscreen_obs) > 0:
        result = pl.concat([drugscreen_obs, non_drugscreen_obs])
    elif len(drugscreen_obs) > 0:
        result = drugscreen_obs
    else:
        result = non_drugscreen_obs

    if len(obs) != len(result):
        logger.warning(
            "Some observations were not drugscreen queries, or did not have two perturbations. They were skipped."
        )

    return result


class DrugscreenTaskAdapter(TaskAdapter):
    """
    A dataset for drugscreen data.

    For drugscreen data, we assume that the biological context is the disease model.
    """

    kind: Literal["drugscreen"] = "drugscreen"

    perturbation_groupby_cols_types: list[Tuple[str, str]] = Field(
        default_factory=lambda: [("inchikey", "<U27"), ("concentration", "float")]
    )
    perturbation_splitting_col: str = Field(default="inchikey")  # generally we want all doses in same split
    batch_groupby_cols: list[str] = Field(default_factory=lambda: ["batch_center"])
    context_groupby_cols: list[str] = Field(default_factory=lambda: ["plate_disease_model", "cell_type"])

    _is_prepared: bool = False
    _filtered_perturbed_obs: pl.DataFrame | None = None
    _filtered_basal_obs: pl.DataFrame | None = None

    def get_all_perturbed_obs(self) -> pl.DataFrame:
        if self._filtered_perturbed_obs is None:
            obs = self.dataset.obs
            obs = obs.filter(pl.col("drugscreen_query"))

            mask = pl.col("perturbations").list.len().eq(2)
            skipped = obs.filter(~mask)
            if len(skipped) > 0:
                logger.warning(
                    f"Some observations had more than two perturbations. They were skipped.\n"
                    f"Path: {self.dataset.obs_path}\n"
                    f"IDs: {skipped['obs_id'].to_list()}"
                )

            obs = obs.filter(mask)

            self._filtered_perturbed_obs = obs

        return self._filtered_perturbed_obs

    def get_all_basal_obs(self) -> pl.DataFrame:
        if self._filtered_basal_obs is None:
            obs = self.dataset.obs.filter(pl.col("is_base_state"))
            self._filtered_basal_obs = obs
        return self._filtered_basal_obs

    def prepare(self) -> None:
        """
        Any (costly) operations that need to be done once on the dataset should be done here.
        """
        if self._is_prepared:
            return

        # Check that the disease model is consistent
        check_disease_model_consistency(self.dataset.obs)

        # Since multiple adapter instances can reference the same dataset,
        # some of the preprocessing here may already have been done, even if _is_prepared is False.
        # Out of precaution, we reset and recompute.
        self.dataset._cached_obs = None

        obs = self.dataset.obs
        obs = obs.with_row_index("original_index")
        obs = add_compound_perturbation_to_obs(obs)
        self.dataset.obs = obs

        self._is_prepared = True
        self._filtered_perturbed_obs = None
        self._filtered_basal_obs = None

    def get_basal_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.get_all_basal_obs().filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]

    def get_perturbations(self, *predictates: pl.Expr) -> np.ndarray:
        # Filter for drugscreen queries
        obs = self.get_all_perturbed_obs().filter(*predictates)

        # Get the perturbations and convert to numpy array
        sortedcols = sorted(self.perturbation_groupby_cols)
        sortedtypes = sorted(self.perturbation_groupby_dtype)
        perturbations = list(obs[sortedcols].iter_rows())
        dt = np.dtype(sortedtypes)

        perturbations = np.array(perturbations, dtype=dt)
        return perturbations

    def get_perturbed_states(self, *predictates: pl.Expr) -> np.ndarray:
        obs = self.get_all_perturbed_obs().filter(*predictates)
        return self.dataset.X[obs["original_index"].to_list()]
