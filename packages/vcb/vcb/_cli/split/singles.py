from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Fold, Split
from vcb.data_models.task.singles import SinglesTaskAdapter
from vcb._cli.split.utils import log_step, POS_CONTROL_FILTER


def singles_split_cli(
    dataset_dir: Path,
    output_dir: Path,
    version: str,
    splitting_level: str = "unique_perturbation",
    splitting_strategy: str = "random",
    validation_ratio: float = 0.15,
    subsample_ratio: float | None = None,
):
    """
    Split a singles dataset into train, validation and test sets.
    """

    # Load the dataset
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=dataset_dir).model_dump())

    task_adapter = SinglesTaskAdapter(dataset=dataset)
    # Preprocess the observations
    task_adapter.prepare()
    original = task_adapter.dataset.obs

    log_step("Original", len(original))

    # Filter out any observations with a length that differs from
    # the task's expectation for either perturbations or controls
    obs = original.filter(task_adapter.perturbation_length_filter)
    log_step("Valid perturbation and control lengths", len(obs))

    # Filter out explicitly labeled positive controls
    obs = obs.filter(POS_CONTROL_FILTER)
    log_step("No positive controls", len(obs))

    # Count base states per disease model
    # Filter out any disease models with less than 100 base states
    FULL_GROUPING = "full_grouping"
    assert FULL_GROUPING not in obs.columns
    obs = obs.with_columns(
        pl.concat_str(task_adapter.context_groupby_cols, separator=":").alias(FULL_GROUPING)
    )

    groups_with_enough_base_states = (
        obs.filter(pl.col("is_negative_control"))
        .group_by(FULL_GROUPING)
        .len(name="count")
        .filter(pl.col("count").ge(100))[FULL_GROUPING]
        .unique()
    )

    obs = obs.filter(pl.col(FULL_GROUPING).is_in(set(groups_with_enough_base_states)))
    log_step("≥100 base states", len(obs))

    # Count unique perturbations per grouped context
    # Filter out any context with less than 25 unique compounds
    perturbations_per_base_state = (
        obs.group_by(FULL_GROUPING)
        .agg(pl.col(task_adapter.perturbation_groupby_cols).n_unique().alias("count"))
        .filter(pl.col("count").ge(25))[FULL_GROUPING]
        .to_list()
    )
    obs = obs.filter(pl.col(FULL_GROUPING).is_in(perturbations_per_base_state))
    log_step("≥25 unique distinct perturbations per base state", len(obs))

    # Sanity check
    # Can we use the index of the subsampled ID to get the row in the original dataframe?
    expected = np.random.randint(0, len(original), size=min(50, len(original))).tolist()
    found = original[expected]["original_index"].to_list()
    assert expected == found, f"Sanity check failed: {expected} != {found}"

    # Filter to only perturbations for final splitting
    # This is de facto borrowing the perturbations filter from task_adapter,
    # and using the batch_center to pull in all the filtering for states with
    # enough critical mass above
    if len(task_adapter.batch_groupby_cols) != 1:
        raise NotImplementedError(
            "current implementation only supports single batch_groupby_cols; this is simple, but not ideal, so let's implement when needed"
        )
    # this takes the first and only element of batch_groupby_cols; if we want to support more: -> .with_columns
    batch_center = task_adapter.batch_groupby_cols[0]

    perturbations_subset = task_adapter.get_all_perturbed_obs().filter(
        pl.col(batch_center).is_in(obs[batch_center].unique())
    )

    # We split randomly on the compound level
    if splitting_level == "observation":
        splitting_col = "original_index"
    elif splitting_level == "unique_perturbation":
        splitting_col = task_adapter.perturbation_splitting_col
    else:
        raise ValueError(
            f"Invalid splitting level: {splitting_level}. Choose from: 'observation', 'unique_peturbation'."
        )

    if splitting_strategy != "random":
        raise ValueError(f"Invalid splitting strategy: {splitting_strategy}. Choose from: 'random'.")

    unique_values = perturbations_subset[splitting_col].unique()
    # maybe make it small to speed things up
    if subsample_ratio is not None:
        subsample_size = int(len(unique_values) * subsample_ratio)
        unique_values = np.random.choice(unique_values, size=subsample_size, replace=False)
        perturbations_subset = perturbations_subset.filter(pl.col(splitting_col).is_in(unique_values))
        log_step("Subsampled", len(perturbations_subset))

    # Find the batch-paired negative controls / base_states
    base_state_indices = (
        original.filter(pl.col("is_negative_control"))
        .filter(pl.col(batch_center).is_in(perturbations_subset[batch_center].unique()))["original_index"]
        .to_list()
    )

    # 5x5 Cross Validation
    folds = []
    for i in range(5):
        random_cv = KFold(n_splits=5, random_state=i, shuffle=True)

        for j, (finetune, test) in enumerate(random_cv.split(unique_values)):
            train_perturbations = unique_values[finetune]
            test_perturbations = unique_values[test]

            if validation_ratio > 0:
                val_size = int(len(train_perturbations) * validation_ratio)
                val_perturbations = np.random.choice(train_perturbations, size=val_size, replace=False)
                train_perturbations = np.setdiff1d(train_perturbations, val_perturbations)
            else:
                val_perturbations = []

            train_subset = perturbations_subset.filter(pl.col(splitting_col).is_in(train_perturbations))
            test_subset = perturbations_subset.filter(pl.col(splitting_col).is_in(test_perturbations))
            validation_subset = perturbations_subset.filter(pl.col(splitting_col).is_in(val_perturbations))

            fold = Fold(
                outer_fold=i,
                inner_fold=j,
                test=test_subset["original_index"].to_list(),
                finetune=train_subset["original_index"].to_list(),
                validation=validation_subset["original_index"].to_list(),
            )
            folds.append(fold)

    # Create the split model
    split_model = Split(
        dataset_id=dataset.dataset_id,
        version=version,
        folds=folds,
        splitting_level=splitting_level,
        splitting_strategy=splitting_strategy,
        controls=base_state_indices,  # there is no difference between controls and base state here
        base_states=base_state_indices,
    )

    # Reload the dataset for a sanity check
    # flagging that the asserts are not following an exceptionally extensible pattern customized from the task adapter
    obs = pl.read_parquet(dataset.obs_path)
    obs = obs.with_row_index("original_index")

    assert all(obs.filter(pl.col("original_index").is_in(split_model.controls))["is_negative_control"])
    assert all(obs.filter(pl.col("original_index").is_in(split_model.base_states))["is_negative_control"])

    # at least one perturbation in the well is a perturbation with usage type "query"
    obs = obs.with_columns(
        pl.col("perturbations")
        .list.eval(pl.element().struct.field("usage_class").eq("query"))
        .list.any()
        .alias("has_query_perturbation")
    )

    for fold in split_model.folds:
        assert all(obs.filter(pl.col("original_index").is_in(fold.finetune))["has_query_perturbation"])
        assert all(obs.filter(pl.col("original_index").is_in(fold.validation))["has_query_perturbation"])
        assert all(obs.filter(pl.col("original_index").is_in(fold.test))["has_query_perturbation"])

    logger.info("\n" + str(split_model))

    output_dir = output_dir / dataset.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"split_{splitting_level}_{splitting_strategy}__v{version}.json"
    output_path = output_dir / fname

    with open(output_path, "w") as fd:
        fd.write(split_model.model_dump_json())

    logger.info(f"Split saved to {output_path}")
