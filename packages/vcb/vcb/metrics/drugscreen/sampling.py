import numpy as np
import polars as pl
from loguru import logger
from sklearn.utils.extmath import _approximate_mode

from vcb.data_models.task.drugscreen import add_compound_perturbation_to_obs
from vcb.metrics.drugscreen.utils import SynchronizedDataset


def compute_glyph_sample_size(obs: pl.DataFrame) -> int:
    """Compute the glyph sample size.

    Given the observations, it determines what the mode of the replicate count is to give
    the proper sampling size to be used in the creation of the glyphs in prometheus space.
    In the event that 2 values are returned by `mode`, the smallest value is returned.

    NOTE: `min` was selected as the tie breaker because it can be thought of as
    something like the least-common-denominator— where the default aggregation
    size will be the smaller of two values of there are two values that are
    equally represented in the experiment.  Importantly, this tie-breaking ONLY
    comes into play when there are multiple group sizes of exactly the same size.
    """
    obs = obs.filter(pl.col("drugscreen_query"))
    obs = add_compound_perturbation_to_obs(obs)

    # NOTE (alisandra): I'm not 100% on if the grouping should be by "experiment_label" or "plate_disease_model"...
    #   I will have to dig in more to exactly how they're designing things.
    #   As larger conceptual experiments in the abstract sense do sometimes get split across multiple RXRX experiment_labels
    #   (set of up to 12 plates seeded together and generally ran in parallel).

    groupby_columns = ["experiment_label", "cell_type", "inchikey", "concentration"]
    sizes = obs.group_by(groupby_columns).len()
    mode = sizes["len"].mode().min()
    logger.debug(f"Distribution of the number of replicates:\n{sizes['len'].describe()}")
    logger.debug(f"Picked the following glyph sample size: {mode}")
    return mode


def get_stratified_sample_counts(classes: list, budget: int, rng: np.random.RandomState | None = None):
    """
    Returns the counts such that we draw a total of `budget` samples,
    trying to spread these samples across the classes such that
    the proportions of the classes are matched as closely as possible.
    """
    if rng is None:
        rng = np.random.RandomState()

    unique_classes, class_sizes = np.unique(classes, return_counts=True)

    # _approximate_mode is a Scikit-learn uitility function that returns
    # the number of samples to draw from each class.
    sample_counts = _approximate_mode(class_sizes, budget, rng)

    # These counts can be zero, which we filter out here.
    sample_classes = np.argwhere(sample_counts).flatten()
    sample_counts = sample_counts[sample_classes]

    return {class_name.item(): count for class_name, count in zip(unique_classes, sample_counts)}


def sample_stratified(
    data: np.ndarray,
    classes: list,
    sample_size: int,
    n_per_sample: int,
    rng: np.random.RandomState | None = None,
    reference_classes: list | None = None,
    replace: bool = False,
):
    """
    Samples data in a stratified manner
    """

    if rng is None:
        rng = np.random.RandomState()

    # Get the stratified sample counts
    if reference_classes is None:
        reference_classes = classes
    sample_counts = get_stratified_sample_counts(reference_classes, n_per_sample, rng)

    unique_classes, inverse, class_sizes = np.unique(classes, return_inverse=True, return_counts=True)
    class_indices = np.split(np.argsort(inverse, kind="mergesort"), np.cumsum(class_sizes)[:-1])

    # Now we sample the data in stratified manner.
    sample_indices = -np.ones((sample_size, n_per_sample), dtype=int)

    # For each sample...
    for i in range(sample_size):
        current_idx = 0

        # For each class that we will sample from...
        for class_name, class_count in sample_counts.items():
            class_idx = np.where(unique_classes == class_name)[0][0]
            replace_current = replace or (class_indices[class_idx].shape[0] < class_count)

            if not replace and replace_current:
                logger.debug(
                    f"For sample {i}, class {class_name} has less samples than the number of samples to draw from it: "
                    f"{len(class_indices[class_idx])} < {class_count}."
                )

            # Sample from this specific class
            choice = rng.choice(class_indices[class_idx], class_count, replace=replace_current)

            # Add the samples to the sample indices
            sample_indices[i, current_idx : current_idx + class_count] = choice
            current_idx += class_count

        # Shuffle the indices
        sample_indices[i, :] = rng.permutation(sample_indices[i, :])

    sample_indices = sample_indices
    return data[sample_indices]


def aggregate_treatment_data(
    data: SynchronizedDataset, groupby_columns: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Sample and aggregate the drugscreen data.

    For the drugscreen data, aggregation is straight-forward. We group by perturbation (inchikey + concentration)
    and aggregate the data using the median (less sensitive to outliers than the mean).
    """
    collect = []
    labels = []

    # We sort the data such that the perturbations are ordered by their concentration.
    for label, group in data.group_by(groupby_columns):
        d = np.median(group.X, axis=0)
        collect.append(d)
        labels.append(label)
    return np.vstack(collect), labels


def sample_and_aggregate(
    data: SynchronizedDataset,
    drugscreen: SynchronizedDataset,
    groupby_columns: list[str],
    sample_sizes: dict[str, int],
    glyph_sample_size: int,
    random_state: int = 0,
) -> np.ndarray:
    """Sample and aggregate the disease and healthy clouds.

    There is a bit of complexity here to try match the distributions we aggregate over
    to the distributions we aggregated over in the treatment data. This is especially important because
    of the curse of dimensionality: In high-dimensions, things are much further apart than in low-dimensions.

    For a given experiment, we thus sample batches in a glyph proportionally
    to the frequency of the batches in the treatment data.

    Assume we want to sample 1000 glyphs for a given experiment.
    1. Divide the 1000 glyphs proportionally to the frequency of compounds in the treatment data.
    2. For each glyph assigned to a given compound, sample from batches proportionally to the frequency of the batches for that compound.
    """
    collect = []

    rng = np.random.RandomState(random_state)

    # Move this out of the loop to avoid recomputing it
    batch_centers = data.obs["batch_center"].to_list()

    for label, group in drugscreen.group_by(groupby_columns):
        if sample_sizes.get(label, 0) == 0:
            continue

        samples = sample_stratified(
            data=data.X,
            classes=batch_centers,
            reference_classes=group.obs["batch_center"].to_list(),
            rng=rng,
            sample_size=sample_sizes[label],
            n_per_sample=glyph_sample_size,
        )
        collect.append(samples)

    data = np.vstack(collect)
    agg_data = np.median(data, axis=1)

    logger.debug(f"Sampled {data.shape} glyphs. Aggregated to {agg_data.shape} using the median.")
    return agg_data
