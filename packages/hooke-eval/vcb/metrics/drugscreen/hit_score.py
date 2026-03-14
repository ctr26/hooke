import numpy as np
from loguru import logger


def compute_hit_scores(
    healthy_cloud: np.ndarray,
    disease_model_cloud: np.ndarray,
    treatment_data: np.ndarray,
    n_sigma_y: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """Compute the hit scores for the drugscreen data. The range is (-1, 1].

    Intuitively, the idea here is to use some sort of weighted average of the principle axis (disease score)
    and the rejection axis (side effect score) to compute the hit score.

    The precise logic behind the below mathematics seems to be lost to history, but it will create these rectangles
    with rounded corners, centered around the origin.
    """
    projection = treatment_data[:, 0]
    rejection = treatment_data[:, 1]

    # Using the standard deviation here is a way to incorporate the separability of the clouds
    # in the hit score calculation. If we didn't do this, we would need to use something like z-factors
    projection_origin_scale = healthy_cloud[:, 0].std()
    projection_target_scale = disease_model_cloud[:, 0].std()

    projection_threshold = np.max(
        [
            projection_origin_scale / (projection_origin_scale + projection_target_scale),
            1 - 3 * projection_target_scale,
        ]
    )

    # Computed a weighted average of the rejection and location coordinate.
    # NOTE (cwognum): This resembles a Generalized Gaussian function, but I don't understand the rationale for the constants.
    rejection_loc = max(healthy_cloud[:, 1].mean(), disease_model_cloud[:, 1].mean())
    rejection_scale = max(healthy_cloud[:, 1].std(), disease_model_cloud[:, 1].std())

    rejection = rejection - rejection_loc

    n_proj = np.log(np.log(0.75) / np.log(0.5)) / np.log(projection_threshold)
    a_proj = np.exp(np.log(-np.log(0.5)) / n_proj)
    a_rej = np.sqrt(-np.log(0.5) / ((rejection_scale * n_sigma_y) ** 2))

    projection = np.clip(projection, a_min=0, a_max=None)
    rejection = np.clip(rejection, a_min=0, a_max=None)

    score = 2 * np.exp(-(abs(a_proj * projection) ** n_proj + (a_rej * rejection) ** 2)) - 1

    if verbose:
        logger.debug(
            f"Distribution of perturbation-level hit scores:\n"
            f"    ≤ 0:         {(score <= 0).sum()} perturbations\n"
            f"    (0.0, 0.25]: {((0 < score) & (score <= 0.25)).sum()} perturbations\n"
            f"    (0.25, 0.5]: {((0.25 < score) & (score <= 0.5)).sum()} perturbations\n"
            f"    (0.5, 0.75]: {((0.5 < score) & (score <= 0.75)).sum()} perturbations\n"
            f"    (0.75, 1.0]: {((0.75 < score) & (score <= 1)).sum()} perturbations"
        )

    return score


def aggregate_hit_scores_per_compound(
    perturbation_scores: np.ndarray, perturbation_labels: np.ndarray
) -> dict[str, float]:
    """Aggregate the hit scores per compound.

    Computes hit scores from aggregates of prometheus projections.

    The hit scores are returned per dose and here aggregated to the compound level by
    taking the mean of 2 highest hit scores (if there are >= 6 doses) or the highest hit score otherwise.
    """
    compounds = np.array([p[0] for p in perturbation_labels])
    unique_compounds = np.unique(compounds)

    scores = []
    for compound in unique_compounds:
        mask = compounds == compound
        compound_scores = perturbation_scores[mask]

        # If there are less than 6 doses, we use the highest score.
        # If there are >= 6 doses, we use the mean of the highest two scores.
        if len(compound_scores) < 6:
            score = np.max(compound_scores)
        else:
            nlargest = np.argpartition(compound_scores, -2)[-2:]
            score = np.mean(compound_scores[nlargest])

        scores.append(score)

    scores = np.array(scores)
    logger.debug(
        f"Distribution of compound-level hit scores:\n"
        f"    ≤ 0:         {(scores <= 0).sum()} compounds\n"
        f"    (0.0, 0.25]: {((0 < scores) & (scores <= 0.25)).sum()} compounds\n"
        f"    (0.25, 0.5]: {((0.25 < scores) & (scores <= 0.5)).sum()} compounds\n"
        f"    (0.5, 0.75]: {((0.5 < scores) & (scores <= 0.75)).sum()} compounds\n"
        f"    (0.75, 1.0]: {((0.75 < scores) & (scores <= 1)).sum()} compounds"
    )

    scores = {k: v for k, v in zip(unique_compounds, scores)}
    return scores
