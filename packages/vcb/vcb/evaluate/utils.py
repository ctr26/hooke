def from_perturbations_to_compound_conc(perturbations: list[dict]) -> str:
    """
    Given a list of perturbations, return the drug compound and concentration.
    For drugscreen data, we can assume that it's the last perturbation in the list.

    If there is no perturbations or the last perturbation is not a compound perturbation, return None.
    """

    if len(perturbations) == 0:
        return None

    sorted_perturbations = sorted(perturbations, key=lambda x: x["hours_post_reference"])

    # Should be fine, but a quick sanity check won't hurt.
    last_perturbation = sorted_perturbations[-1]
    if last_perturbation["type"] != "compound":
        return None

    # standardize concentration
    # to account for differences in precision and rounding
    conc_formatted = f"{last_perturbation['concentration']:.3e}"

    # create hashable tuple
    return (last_perturbation["inchikey"], conc_formatted)
