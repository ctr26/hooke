import numpy as np
import polars as pl
from loguru import logger


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


def from_perturbations_to_compound(perturbations: list[dict]) -> str:
    """
    Given a list of perturbations, return the drug compound.
    For drugscreen data, we can assume that it's the last perturbation in the list.

    If there is no perturbations or the last perturbation is not a compound perturbation, return None.
    """

    if len(perturbations) == 0:
        return None

    sorted_perturbations = sorted(
        perturbations, key=lambda x: x["hours_post_reference"]
    )

    # Should be fine, but a quick sanity check won't hurt.
    last_perturbation = sorted_perturbations[-1]
    if last_perturbation["type"] != "compound":
        return None

    return last_perturbation["inchikey"]


def add_compound_perturbation_to_obs(
    obs: pl.DataFrame, assume_only_two_perturbations: bool = True
) -> pl.DataFrame:
    """
    Add the compound perturbation to the observations, including the inchikey and concentration.
    """

    # For robustness, ensure we're only working with drugscreen queries
    assert obs["drugscreen_query"].all(), (
        "Some observations were not drugscreen queries"
    )

    # Not all drugscreen queries necessarily have two perturbations.
    # We expect all predictions to have two perturbations, but the ground truth may not.
    only_two_perturbations = obs["perturbations"].list.len() == 2
    if not assume_only_two_perturbations:
        obs = obs.filter(only_two_perturbations)
    else:
        assert only_two_perturbations.all(), (
            "Some observations did not have two perturbations"
        )

    # explode = flatten list of perturbation
    # unnest = turn dict/struct into columns
    col = (
        obs.explode("perturbations")
        .unnest("perturbations")
        .filter(pl.col("inchikey").is_not_null())
        .select("inchikey", "concentration")
    )

    if len(obs) != len(obs):
        logger.warning(
            "Some observations were not drugscreen queries, or did not have two perturbations. They were skipped."
        )

    return obs.with_columns(col)


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
