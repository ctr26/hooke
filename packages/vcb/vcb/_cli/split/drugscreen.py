from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import KFold

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Fold, Split


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


def log_step(step: str, count: int) -> None:
    """
    Log the summary of the drugscreen split with aligned formatting.
    """
    # Format with fixed width for alignment (adjust width as needed)
    logger.info(f"{step:<35} {count:>8,}")


def drugscreen_split_cli(
    dataset_dir: Path,
    output_dir: Path,
    version: str,
    splitting_level: str = "compound",
    splitting_strategy: str = "random",
    validation_ratio: float = 0.15,
    subsample_ratio: float | None = None,
):
    """
    Split a drugscreen dataset into a train and test set.
    """

    # Load the dataset
    dataset = AnnotatedDataMatrix.from_dataset_directory(DatasetDirectory(root=dataset_dir))

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

    # Sanity check
    # Can we use the index of the subsampled ID to get the row in the original dataframe?
    expected = np.random.randint(0, len(original), size=50).tolist()
    found = original[expected]["original_index"].to_list()
    assert expected == found, f"Sanity check failed: {expected} != {found}"

    # Find the batch-paired negative controls
    negative_control_indices = (
        original.filter(pl.col("is_negative_control"))
        .filter(pl.col("batch_center").is_in(obs["batch_center"].unique()))["original_index"]
        .to_list()
    )

    base_state_indices = (
        original.filter(pl.col("is_base_state"))
        .filter(pl.col("batch_center").is_in(obs["batch_center"].unique()))["original_index"]
        .to_list()
    )

    drugscreen_subset = add_compound(
        original.filter(pl.col("drugscreen_query")).filter(
            pl.col("batch_center").is_in(obs["batch_center"].unique())
        )
    )

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
        unique_values = np.random.choice(unique_values, size=subsample_size, replace=False)

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

    assert all(obs.filter(pl.col("original_index").is_in(split_model.controls))["is_negative_control"])
    assert all(obs.filter(pl.col("original_index").is_in(split_model.base_states))["is_base_state"])

    for fold in split_model.folds:
        assert all(obs.filter(pl.col("original_index").is_in(fold.finetune))["drugscreen_query"])
        assert all(obs.filter(pl.col("original_index").is_in(fold.validation))["drugscreen_query"])
        assert all(obs.filter(pl.col("original_index").is_in(fold.test))["drugscreen_query"])

    logger.info("\n" + str(split_model))

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{dataset.dataset_id}__split_{splitting_level}_{splitting_strategy}__v{version}.json"
    with open(output_dir / fname, "w") as fd:
        fd.write(split_model.model_dump_json())

    logger.info(f"Split saved to {output_dir / fname}")
