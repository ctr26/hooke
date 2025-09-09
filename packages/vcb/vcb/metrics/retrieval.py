import numpy as np
from tqdm import tqdm

from vcb.metrics.distributional.mmd import compute_e_distance
from vcb.models.misc import CompoundPerturbation


def calculate_grouping_stats(
    samples_truth: np.ndarray,
    group_labels_truth: list[CompoundPerturbation],
    unique_groups: list[CompoundPerturbation],
):
    stats = {
        "group_size": [],
        "intra_group_variance": [],
    }

    for group in unique_groups:
        stats["group_size"].append(len(group))
        stats["intra_group_variance"].append(
            np.var(samples_truth[group_labels_truth == group])
        )

    return {f"retrieval_mean_{k}": np.mean(v) for k, v in stats.items()}


def from_similarities_to_retrieval_score(similarities: np.ndarray) -> float:
    # get ranks of true sample for each generated sample
    # sort from lowest to highest distance
    ranks_argsort = np.argsort(similarities, axis=1)
    ranks_indicator = ranks_argsort == np.arange(ranks_argsort.shape[0]).reshape(-1, 1)

    # extract rank of correct pairing
    ranks = np.nonzero(ranks_indicator)[1]

    # normalize by number of comparisons
    score = 1 - np.mean(ranks) / (similarities.shape[0] - 1)
    return score


def calculate_mae_retrieval(
    samples_pred: np.ndarray,
    samples_truth: np.ndarray,
    group_labels_pred: list[CompoundPerturbation],
    group_labels_truth: list[CompoundPerturbation],
    suffix: str = "_mae",
) -> dict[str, float]:
    """
    Calculate MAE-based retrieval for a given set of generated samples against a groundtruth.

    Computes the MAE between the average features of the predicted samples and the average features of the ground truth samples.
    """

    dt = np.dtype([("inchikey", "U27"), ("concentration", float)])
    group_labels_pred = np.array(group_labels_pred, dtype=dt)
    group_labels_truth = np.array(group_labels_truth, dtype=dt)

    # Get the groups to compare
    unique_groups = np.intersect1d(
        np.unique(group_labels_pred),
        np.unique(group_labels_truth),
    )

    group_means_pred = {}
    group_means_truth = {}

    for group in unique_groups:
        # Get indices for this group
        pred_mask = group_labels_pred == group
        truth_mask = group_labels_truth == group

        # Convert group to hashable tuple for dictionary key
        group_key = (group["inchikey"], group["concentration"])

        # Compute mean sample for each group
        pred_samples = samples_pred[pred_mask]
        group_means_pred[group_key] = np.mean(pred_samples, axis=0)

        truth_samples = samples_truth[truth_mask]
        group_means_truth[group_key] = np.mean(truth_samples, axis=0)

    # Fully vectorized computation of similarity matrix
    # Stack all mean samples into matrices for vectorized operations
    pred_samples_matrix = np.array(
        [
            group_means_pred[(group["inchikey"], group["concentration"])]
            for group in unique_groups
        ]
    )
    truth_samples_matrix = np.array(
        [
            group_means_truth[(group["inchikey"], group["concentration"])]
            for group in unique_groups
        ]
    )

    pred_expanded = pred_samples_matrix[:, np.newaxis, :]
    truth_expanded = truth_samples_matrix[np.newaxis, :, :]

    sims = np.mean(np.abs(pred_expanded - truth_expanded), axis=2)
    sims = -sims

    stats = calculate_grouping_stats(samples_truth, group_labels_truth, unique_groups)

    # normalize by number of comparisons
    return {
        f"retrieval{suffix}": from_similarities_to_retrieval_score(sims),
        **stats,
    }


def calculate_edistance_retrieval(
    samples_pred: np.ndarray,
    samples_truth: np.ndarray,
    group_labels_pred: list[CompoundPerturbation],
    group_labels_truth: list[CompoundPerturbation],
    suffix: str = "_edistance",
) -> dict[str, float]:
    """
    Calculate normalized retrieval for a given set of generated samples against a groundtruth.

    Computes the E-distance between the predicted samples and the ground truth samples.

    NOTE: We are using E-distance to stay consistent with Cellflow evaluation, but we could also use MMD with a linear/rbf kernel.
    They should be equivalent for the euclidean distance kernel and E-distance might be simpler choice overall, but let's keep this in mind.
    """
    dt = np.dtype([("inchikey", "U27"), ("concentration", float)])
    group_labels_pred = np.array(group_labels_pred, dtype=dt)
    group_labels_truth = np.array(group_labels_truth, dtype=dt)

    # Get the groups to compare
    unique_groups = np.intersect1d(
        np.unique(group_labels_pred),
        np.unique(group_labels_truth),
    )

    n_groups = len(unique_groups)
    sims = np.zeros((n_groups, n_groups))

    for ix1, group1 in tqdm(
        enumerate(unique_groups),
        leave=False,
        desc="E-distance based retrieval",
        total=n_groups,
    ):
        for ix2, group2 in enumerate(unique_groups):
            dist = compute_e_distance(
                samples_pred[group_labels_pred == group1],
                samples_truth[group_labels_truth == group2],
            )
            sims[ix1, ix2] = dist

    stats = calculate_grouping_stats(samples_truth, group_labels_truth, unique_groups)

    # normalize by number of comparisons
    return {f"retrieval{suffix}": from_similarities_to_retrieval_score(sims), **stats}
