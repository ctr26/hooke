from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Fold, Split
from vcb._cli.split.utils import log_step


def add_disease_model(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Add the disease model column to the observations.
    """

    return obs.with_columns(
        disease_model=pl.col("perturbations")
        .list.get(0)
        .pipe(
            lambda expr: pl.when(expr.struct.field("type") == "genetic")
            .then(expr.struct.field("ensembl_gene_id"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )
    )


def add_compound(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Add the disease model column to the observations.
    """

    return obs.with_columns(
        compound=pl.col("perturbations")
        .list.get(1)
        .pipe(
            lambda expr: pl.when(expr.struct.field("type") == "compound")
            .then(expr.struct.field("inchikey"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )
    )


def drugscreen_split_cli(
    dataset_dir: Path,
    output_dir: Path,
    version: str,
    splitting_level: str = "compound",
    splitting_strategy: str = "random",
    validation_ratio: float = 0.15,
    subsample_ratio: float | None = None,
    test_obs_ids_file: Path | None = None,
):
    """
    Split a drugscreen dataset into a train and test set.
    """

    # Convert None to empty list for simpler code
    if test_obs_ids_file is not None:
        test_obs_ids = pl.read_csv(test_obs_ids_file, has_header=False)["column_1"].to_list()
    else:
        test_obs_ids = []

    # Load the dataset
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=dataset_dir).model_dump())

    # Preprocess the observations
    original = dataset.obs.with_row_index("original_index")
    original = original.with_columns(
        pl.col("perturbations").list.eval(
            pl.element().sort_by(pl.element().struct.field("hours_post_reference"))
        )
    )
    log_step("Original", len(original))

    # Filter out the empties (0 perturbations)
    # Filter out any observations with 3 perturbations (probably positive controls)
    obs = original.filter(pl.col("perturbations").list.len().is_in([1, 2]))
    log_step("1 <= Perturbation length <= 2", len(obs))

    # Filter out explicitly labeled positive controls
    obs = obs.filter(
        ~pl.col("perturbations")
        .list.eval(pl.element().struct.field("usage_class").eq("positive_control"))
        .list.any()
    )
    log_step("No positive controls", len(obs))

    # Add the disease model column, as well as the compound perturbation column
    # When sorted by hours_post_reference, the first perturbation should be the disease model
    obs = add_disease_model(obs)

    # Count base states per disease model
    # Filter out any disease models with less than 100 base states
    disease_models_with_enough_base_states = (
        obs.filter(pl.col("is_base_state"))
        .group_by("disease_model")
        .len(name="count")
        .filter(pl.col("count").ge(100))["disease_model"]
        .unique()
    )
    obs = obs.filter(pl.col("disease_model").is_in(disease_models_with_enough_base_states))
    log_step("≥100 base states", len(obs))

    # Count unique compounds per disease model
    # Filter out any disease models with less than 25 unique compounds
    compounds_per_disease_model = (
        add_compound(obs.filter(pl.col("drugscreen_query")))
        .group_by("disease_model")
        .agg(pl.col("compound").n_unique().alias("count"))
        .filter(pl.col("count").ge(25))["disease_model"]
        .to_list()
    )
    obs = obs.filter(pl.col("disease_model").is_in(compounds_per_disease_model))
    log_step("≥25 unique compounds", len(obs))

    # Filter out any observations that are not drugscreen queries
    obs = obs.filter(pl.col("drugscreen_query"))
    log_step("Is drugscreen query", len(obs))

    # Sanity check
    # Can we use the index of the subsampled ID to get the row in the original dataframe?
    expected = np.random.randint(0, len(original), size=min(50, len(original))).tolist()
    found = original[expected]["original_index"].to_list()
    assert expected == found, f"Sanity check failed: {expected} != {found}"

    drugscreen_subset = add_compound(obs)

    # We split randomly on the compound level
    if splitting_level == "observation":
        splitting_col = "original_index"
    elif splitting_level == "compound":
        splitting_col = "compound"
    else:
        raise ValueError(
            f"Invalid splitting level: {splitting_level}. Choose from: 'observation', 'compound'."
        )

    if splitting_strategy != "random":
        raise ValueError(f"Invalid splitting strategy: {splitting_strategy}. Choose from: 'random'.")

    unique_values = drugscreen_subset[splitting_col].unique()
    if subsample_ratio is not None:
        subsample_size = int(len(unique_values) * subsample_ratio)
        unique_values = np.random.choice(unique_values.to_numpy(), size=subsample_size, replace=False)
        drugscreen_subset = drugscreen_subset.filter(pl.col(splitting_col).is_in(unique_values))
        log_step("Subsampled", len(drugscreen_subset))
    else:
        unique_values = unique_values.to_numpy()

    # Handle explicitly specified test observation IDs
    test_splitting_values = np.array([], dtype=unique_values.dtype)
    if len(test_obs_ids) > 0:
        # Find which observations from test_obs_ids exist in drugscreen_subset
        test_obs_filtered = drugscreen_subset.filter(pl.col("obs_id").is_in(test_obs_ids))
        found_test_obs_ids = set(test_obs_filtered["obs_id"].to_list())
        missing_test_obs_ids = set(test_obs_ids) - found_test_obs_ids

        if missing_test_obs_ids:
            logger.warning(
                f"Some test_obs_ids were not found in the filtered drugscreen subset: "
                f"{missing_test_obs_ids}. These will be ignored."
            )

        if len(found_test_obs_ids) > 0:
            # Extract the splitting values for these test observations
            test_splitting_values = test_obs_filtered[splitting_col].unique().to_numpy()

            # sanity check consistency between splitting level and test_obs_ids
            not_test_obs_filtered = drugscreen_subset.filter(~pl.col("obs_id").is_in(test_obs_ids))
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

    # Find the batch-paired negative controls
    negative_control_indices = (
        original.filter(pl.col("is_negative_control"))
        .filter(pl.col("batch_center").is_in(drugscreen_subset["batch_center"].unique()))["original_index"]
        .to_list()
    )

    base_state_indices = (
        original.filter(pl.col("is_base_state"))
        .filter(pl.col("batch_center").is_in(drugscreen_subset["batch_center"].unique()))["original_index"]
        .to_list()
    )

    # 5x5 Cross Validation
    folds = []
    for i in range(5):
        random_cv = KFold(n_splits=5, random_state=i, shuffle=True)

        for j, (finetune, test) in enumerate(random_cv.split(unique_values)):
            train_perturbations = unique_values[finetune]
            test_perturbations = unique_values[test]

            # Always add explicitly specified test splitting values to test set
            if len(test_splitting_values) > 0:
                test_perturbations = np.concatenate([test_perturbations, test_splitting_values])

            if validation_ratio > 0:
                val_size = int(len(train_perturbations) * validation_ratio)
                val_perturbations = np.random.choice(train_perturbations, size=val_size, replace=False)
                train_perturbations = np.setdiff1d(train_perturbations, val_perturbations)
            else:
                val_perturbations = []

            train_subset = drugscreen_subset.filter(pl.col(splitting_col).is_in(train_perturbations))
            test_subset = drugscreen_subset.filter(pl.col(splitting_col).is_in(test_perturbations))
            validation_subset = drugscreen_subset.filter(pl.col(splitting_col).is_in(val_perturbations))

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
        controls=negative_control_indices,
        base_states=base_state_indices,
    )

    # Reload the dataset for a sanity check
    obs = pl.read_parquet(dataset.obs_path)
    obs = obs.with_row_index("original_index")
    obs = obs.with_columns(
        pl.col("perturbations").list.eval(
            pl.element().sort_by(pl.element().struct.field("hours_post_reference"))
        )
    )

    controls = obs.filter(pl.col("original_index").is_in(split_model.controls))
    assert all(controls["is_negative_control"])
    assert all(controls["perturbations"].list.len().is_in([0, 1]))

    base_states = obs.filter(pl.col("original_index").is_in(split_model.base_states))
    assert all(base_states["is_base_state"])
    assert all(base_states["perturbations"].list.len().eq(1))

    for fold in split_model.folds:
        finetune = obs.filter(pl.col("original_index").is_in(fold.finetune))
        validation = obs.filter(pl.col("original_index").is_in(fold.validation))
        test = obs.filter(pl.col("original_index").is_in(fold.test))

        assert all(finetune["drugscreen_query"])
        assert all(validation["drugscreen_query"])
        assert all(test["drugscreen_query"])

        assert all(finetune["perturbations"].list.len().eq(2))
        assert all(validation["perturbations"].list.len().eq(2))
        assert all(test["perturbations"].list.len().eq(2))

        if splitting_level == "compound":
            finetune_compounds = set(add_compound(finetune)["compound"].unique())
            validation_compounds = set(add_compound(validation)["compound"].unique())
            test_compounds = set(add_compound(test)["compound"].unique())

            assert finetune_compounds.intersection(validation_compounds) == set(), (
                f"Finetune and validation have {finetune_compounds.intersection(validation_compounds)}"
            )
            assert finetune_compounds.intersection(test_compounds) == set(), (
                f"Finetune and test have {finetune_compounds.intersection(test_compounds)}"
            )
            assert validation_compounds.intersection(test_compounds) == set(), (
                f"Validation and test have {validation_compounds.intersection(test_compounds)}"
            )

    # Validate that test_obs_ids are always in test sets and never in train/validation
    if len(test_obs_ids) > 0:
        found_test_indices = set(
            drugscreen_subset.filter(pl.col("obs_id").is_in(test_obs_ids))["original_index"].to_list()
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
                        f"Fold ({fold.outer_fold}, {fold.inner_fold}): Some test_obs_ids are missing from test set: "
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
                if splitting_level != "observation":
                    test_obs_in_fold = drugscreen_subset.filter(
                        pl.col("original_index").is_in(list(found_test_indices))
                    )
                    test_splitting_vals = set(test_obs_in_fold[splitting_col].unique().to_list())

                    # Find all observations with these splitting values
                    all_obs_with_test_splitting_vals = drugscreen_subset.filter(
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
