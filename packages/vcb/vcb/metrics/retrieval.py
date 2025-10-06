import numpy as np
from tqdm import tqdm

from vcb.data_models.misc import CompoundPerturbation
from vcb.metrics.distributional.mmd import compute_e_distance


def calculate_grouping_stats(
    samples_truth: np.ndarray,
    group_labels_truth: list[CompoundPerturbation],
    unique_groups: list[CompoundPerturbation],
):
    stats = {
        "population_size": [],
        "intra_population_variance": [],
    }

    for group in unique_groups:
        population = samples_truth[group_labels_truth == group]
        stats["population_size"].append(len(population))
        stats["intra_population_variance"].append(np.var(population))

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
    y_pred: np.ndarray,
    y_true: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
    y_base: np.ndarray | None = None,
    use_deltas: bool = False,
) -> tuple[float, dict]:
    # Get the groups to compare
    unique_groups = np.intersect1d(np.unique(p_true), np.unique(p_pred))

    group_means_pred = {}
    group_means_truth = {}

    if use_deltas:
        # TODO (cwognum): Since we subtract a constant vector, computing retrieval with or without deltas should be the same.
        #   This is not what we want. We need to change how we compute the deltas here (i.e. how do we pair base and perturbed samples?)
        y_base_mean = np.mean(y_base, axis=0)
        samples_true = y_true - y_base_mean
        samples_pred = y_pred - y_base_mean
    else:
        samples_true = y_true
        samples_pred = y_pred

    for group in unique_groups:
        # Get mask for this group
        pred_mask = p_pred == group
        truth_mask = p_true == group

        # Convert group to hashable tuple for dictionary key
        group_key = tuple(group)

        # Compute mean sample for each group
        pred_samples = samples_pred[pred_mask]
        group_means_pred[group_key] = np.mean(pred_samples, axis=0)

        truth_samples = samples_true[truth_mask]
        group_means_truth[group_key] = np.mean(truth_samples, axis=0)

    # Fully vectorized computation of similarity matrix
    # Stack all mean samples into matrices for vectorized operations
    pred_samples_matrix = np.array([group_means_pred[tuple(group)] for group in unique_groups])
    truth_samples_matrix = np.array([group_means_truth[tuple(group)] for group in unique_groups])

    pred_expanded = pred_samples_matrix[:, np.newaxis, :]
    truth_expanded = truth_samples_matrix[np.newaxis, :, :]

    sims = np.mean(np.abs(pred_expanded - truth_expanded), axis=2)
    sims = -sims

    stats = calculate_grouping_stats(samples_true, p_true, unique_groups)
    return from_similarities_to_retrieval_score(sims), stats


def calculate_edistance_retrieval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_base: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
) -> tuple[float, dict]:
    """
    Calculate normalized retrieval for a given set of generated samples against a groundtruth.

    Computes the E-distance between the predicted samples and the ground truth samples.

    NOTE: We are using E-distance to stay consistent with Cellflow evaluation, but we could also use MMD with a linear/rbf kernel.
    They should be equivalent for the euclidean distance kernel and E-distance might be simpler choice overall, but let's keep this in mind.
    """
    # Get the groups to compare
    unique_groups = np.intersect1d(np.unique(p_pred), np.unique(p_true))

    n_groups = len(unique_groups)
    sims = np.zeros((n_groups, n_groups))

    y_base_mean = np.mean(y_base, axis=0)
    delta_true = y_true - y_base_mean
    delta_pred = y_pred - y_base_mean

    for ix1, group1 in tqdm(
        enumerate(unique_groups),
        leave=False,
        desc="E-distance based retrieval",
        total=n_groups,
    ):
        for ix2, group2 in enumerate(unique_groups):
            dist = compute_e_distance(
                delta_true[p_true == group1],
                delta_pred[p_pred == group2],
            )
            sims[ix1, ix2] = dist

    supp = calculate_grouping_stats(y_true, p_true, unique_groups)
    return from_similarities_to_retrieval_score(sims), supp
