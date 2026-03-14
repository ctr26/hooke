import json
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold
import typer

from vcb._cli.split.utils import POS_CONTROL_FILTER, log_step
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Fold, Split
from vcb.data_models.task.singles import SinglesTaskAdapter


def _parse_perturbation_groupby_cols_types(value: str | None) -> list[tuple[str, str]] | None:
    """Parse perturbation_groupby_cols_types from JSON string."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(
                "perturbation_groupby_cols_types must be a json string with a list of lists with pairs of [col_name, col_type]"
            )
        return [tuple(item) for item in parsed]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for perturbation_groupby_cols_types: {e}")


def singles_split_cli(
    dataset_dir: Path,
    output_dir: Path,
    version: str,
    splitting_level: str = "unique_perturbation",
    splitting_strategy: str = "random",
    validation_ratio: float = 0.15,
    n_folds: Annotated[
        int,
        typer.Option(
            "--n-folds", help="Number of folds for KFold, this controls the test set fraction (1/n_folds)"
        ),
    ] = 5,
    subsample_ratio: float | None = None,
    test_obs_ids_file: Annotated[
        Path | None,
        typer.Option(
            "--test-obs-ids-file",
            help="Path to file containing test observation IDs, one per line, no header",
        ),
    ] = None,
    perturbation_groupby_cols_types: Annotated[
        str | None,
        typer.Option(
            "--perturbation-groupby-cols-types",
            help='JSON string specifying perturbation_groupby_cols_types, e.g., \'[["inchikey", "<U27"], ["concentration", "float"]]\'',
        ),
    ] = None,
    perturbation_splitting_col: Annotated[
        str | None,
        typer.Option(
            "--perturbation-splitting-col",
            help="Column name to use for perturbation-level splitting (e.g., 'inchikey')",
        ),
    ] = None,
):
    """
    Split a singles dataset into train, validation and test sets.
    """
    if test_obs_ids_file is not None:
        test_obs_ids = pl.read_csv(test_obs_ids_file, has_header=False)["column_1"].to_list()
    else:
        test_obs_ids = []

    # Build task adapter kwargs from config file and CLI (CLI takes precedence)
    task_adapter_kwargs = {}

    # perturbation_groupby_cols_types: from config file or CLI
    if perturbation_groupby_cols_types is not None:
        task_adapter_kwargs["perturbation_groupby_cols_types"] = _parse_perturbation_groupby_cols_types(
            perturbation_groupby_cols_types
        )

    # perturbation_splitting_col: from config file or CLI
    if perturbation_splitting_col is not None:
        task_adapter_kwargs["perturbation_splitting_col"] = perturbation_splitting_col
    # Load the dataset
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=dataset_dir).model_dump())

    task_adapter = SinglesTaskAdapter(dataset=dataset, **task_adapter_kwargs)
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
        .agg(pl.struct(task_adapter.perturbation_groupby_cols).n_unique().alias("count_upert"))
        .filter(pl.col("count_upert").ge(25))[FULL_GROUPING]
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

    perturbations_subset = task_adapter.all_perturbed_obs.filter(
        pl.col(batch_center).is_in(obs[batch_center].unique())
    ).filter(POS_CONTROL_FILTER)

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

    # check if there are any None values in the splitting column, report the obs_id
    none_values = perturbations_subset.filter(pl.col(splitting_col).is_null())["obs_id"].to_list()
    if len(none_values) > 0:
        raise ValueError(
            f"There are {len(none_values)} None values in the splitting column: {splitting_col}, e.g. obs_ids: {none_values[:5]}..."
        )

    unique_values = perturbations_subset[splitting_col].unique()
    # maybe make it small to speed things up
    if subsample_ratio is not None:
        subsample_size = int(len(unique_values) * subsample_ratio)
        unique_values = np.random.choice(unique_values.to_numpy(), size=subsample_size, replace=False)
        perturbations_subset = perturbations_subset.filter(pl.col(splitting_col).is_in(unique_values))
        log_step("Subsampled", len(perturbations_subset))
    else:
        unique_values = unique_values.to_numpy()

    # Handle explicitly specified test observation IDs
    test_splitting_values = np.array([], dtype=unique_values.dtype)
    if len(test_obs_ids) > 0:
        # Find which observations from test_obs_ids exist in perturbations_subset
        test_obs_filtered = perturbations_subset.filter(pl.col("obs_id").is_in(test_obs_ids))
        found_test_obs_ids = set(test_obs_filtered["obs_id"].to_list())
        missing_test_obs_ids = set(test_obs_ids) - found_test_obs_ids

        if missing_test_obs_ids:
            logger.warning(
                f"Some test_obs_ids were not found in the filtered perturbations subset: "
                f"{missing_test_obs_ids}. These will be ignored."
            )

        if len(found_test_obs_ids) > 0:
            # Extract the splitting values for these test observations
            test_splitting_values = test_obs_filtered[splitting_col].unique().to_numpy()

            # sanity check consistency between splitting level and test_obs_ids
            not_test_obs_filtered = perturbations_subset.filter(~pl.col("obs_id").is_in(test_obs_ids))
            not_test_splitting_values = not_test_obs_filtered[splitting_col].unique().to_numpy()
            if set(not_test_splitting_values) & set(test_splitting_values):
                raise ValueError(
                    f"The splitting values inside vs outside the test_obs_ids are not mutually exclusive. "
                    f"This probably indicates that the test_obs_ids were split with logic "
                    f"that does not group by the splitting_level as expected for this split. "
                    f"The overlap is: {set(not_test_splitting_values) & set(test_splitting_values)}"
                )
            # Remove test splitting values from unique_values to exclude them from KFold
            unique_values = np.setdiff1d(unique_values, test_splitting_values)

            logger.info(
                f"Explicitly assigning {len(test_splitting_values)} splitting value(s) to test set "
                f"(from {len(found_test_obs_ids)} test observation IDs)"
            )

    # Find the batch-paired negative controls / base_states
    base_state_indices = (
        original.filter(pl.col("is_negative_control"))
        .filter(pl.col(batch_center).is_in(perturbations_subset[batch_center].unique()))["original_index"]
        .to_list()
    )

    # 5x5 Cross Validation
    folds = []
    for i in range(5):
        random_cv = KFold(n_splits=n_folds, random_state=i, shuffle=True)

        for j, (finetune, test) in enumerate(random_cv.split(unique_values)):
            train_perturbations = unique_values[finetune]
            test_perturbations = unique_values[test]

            # add the splitting values of any explicitly specified test obs_id values to test set
            if len(test_splitting_values) > 0:
                test_perturbations = np.concatenate([test_perturbations, test_splitting_values])

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

    # Validate that test_obs_ids are always in test sets and never in train/validation
    if len(test_obs_ids) > 0:
        found_test_indices = set(
            perturbations_subset.filter(pl.col("obs_id").is_in(test_obs_ids))["original_index"].to_list()
        )

        if len(found_test_indices) > 0:
            for fold in split_model.folds:
                fold_test_indices = set(fold.test)
                fold_train_indices = set(fold.finetune)
                fold_val_indices = set(fold.validation)

                # Check that all test_obs_ids are in test
                missing_from_test = found_test_indices - fold_test_indices
                if missing_from_test:
                    logger.warning(
                        f"Fold ({fold.outer_fold}, {fold.inner_fold}): Some test_obs_ids are missing from test set, indices: "
                        f"{len(missing_from_test)} missing: {list(missing_from_test)[:5]}..."
                    )

                # Check that test_obs_ids are not in train or validation
                in_train = found_test_indices & fold_train_indices
                in_val = found_test_indices & fold_val_indices
                if in_train or in_val:
                    logger.warning(
                        f"Fold ({fold.outer_fold}, {fold.inner_fold}): Some test_obs_ids appear in train/validation, indices: "
                        f"train={len(in_train)}: {list(in_train)[:5]}..., validation={len(in_val)}: {list(in_val)[:5]}..."
                    )

                # When splitting_level != "observation", validate that all observations with the same
                # splitting value as test_obs_ids are also in test
                # e.g. if this is a compound-singles-split and as much as a single well with Comp1 is in the test set
                #  then Comp1 is always and only in the test set
                if splitting_level != "observation":
                    test_obs_in_fold = perturbations_subset.filter(
                        pl.col("original_index").is_in(list(found_test_indices))
                    )
                    test_splitting_vals = set(test_obs_in_fold[splitting_col].unique().to_list())

                    # Find all observations with these splitting values
                    all_obs_with_test_splitting_vals = perturbations_subset.filter(
                        pl.col(splitting_col).is_in(list(test_splitting_vals))
                    )
                    all_obs_ids_with_test_splitting_vals = set(
                        all_obs_with_test_splitting_vals["original_index"].to_list()
                    )

                    # Check if any are in train or validation
                    in_train = all_obs_ids_with_test_splitting_vals & fold_train_indices
                    in_val = all_obs_ids_with_test_splitting_vals & fold_val_indices

                    if in_train or in_val:
                        logger.warning(
                            f"Fold ({fold.outer_fold}, {fold.inner_fold}): Violation of splitting-level constraint! "
                            f"When splitting_level='{splitting_level}', all observations sharing the same splitting "
                            f"value as test_obs_ids must be in test. However, {len(in_train)} observation(s) with "
                            f"test splitting values are in train and {len(in_val)} are in validation. "
                            f"This probably indicates that the test_obs_ids were split with logic "
                            f"that does not group by the splitting_level as expected."
                        )

    logger.info("\n" + str(split_model))

    output_dir = output_dir / dataset.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = f"split_{splitting_level}_{splitting_strategy}__v{version}.json"
    output_path = output_dir / fname

    with open(output_path, "w") as fd:
        fd.write(split_model.model_dump_json())

    logger.info(f"Split saved to {output_path}")
