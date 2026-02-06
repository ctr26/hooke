from pathlib import Path
import polars as pl
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)

INTERNAL_BENCHMARKING_BASE_PATH = Path(
    "/rxrx/data/valence/internal_benchmarking/vcds1/"
)

CROSS_CELL_LINE_BASE_PATH = Path(
    "/rxrx/data/valence/phenomics/cross_cell_line__brightfield__pretrain__v1_0"
)
CROSS_CELL_LINE_OBS_PATH = (
    CROSS_CELL_LINE_BASE_PATH
    / "cross_cell_line__brightfield__pretrain__v1_0_obs.parquet"
)

VCB_OBS_PATH = (
    INTERNAL_BENCHMARKING_BASE_PATH
    / "drugscreen__cell_paint__v1_2"
    / "drugscreen__cell_paint__v1_2_obs.parquet"
)
TEST_OBS_PATH = (
    INTERNAL_BENCHMARKING_BASE_PATH
    / "cross_cell_line__brightfield__v1_0"
    / "cross_cell_line__brightfield__v1_0_obs.parquet"
)

DART_OBS_PATH = Path(
    "/rxrx/data/valence/internal_benchmarking/context_vcds1/dart_example__v1_0/dart_example__v1_0_obs.parquet"
)

COMMON_COLUMNS = [
    "cell_type",
    "concentration",
    "experiment_label",
    "image_path",
    "image_type",
    "order_id",
    "plate_order_read_id",
    "read_id",
    "perturbations",
    "rec_id",
    "shape",
    "split",
    "well_address",
    "source",
]


@dataclass(frozen=True)
class SplitIndices:
    finetune: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def load_vcb_split_indices(splits_path: Path) -> tuple[Dict[str, Any], SplitIndices]:
    """Load finetune/validation/test split indices."""
    with splits_path.open() as handle:
        splits_json = json.load(handle)
    first_fold = splits_json["folds"][0]
    split_indices = SplitIndices(
        finetune=np.array(first_fold["finetune"]),
        validation=np.array(first_fold["validation"]),
        test=np.array(first_fold["test"]),
    )
    return splits_json, split_indices


# Load the VCB splits
SPLITS_PATH = Path(
    "/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json"
)
_, vcb_splits = load_vcb_split_indices(SPLITS_PATH)


def add_columns(
    path: Path,
    image_type: str,
    source: str,
    image_shape: tuple[int, int, int],
    split: str | None = None,
) -> pl.DataFrame:
    df = pl.read_parquet(str(path)).with_columns(
        pl.lit(image_type).alias("image_type"),
        pl.lit(source).alias("source"),
        pl.lit(image_shape).alias("shape"),
        rec_id=pl.col("perturbations").list.eval(
            pl.element().struct.field("source_id")
        ),
        concentration=pl.col("perturbations").list.eval(
            pl.element().struct.field("concentration").cast(pl.Float64).cast(pl.String)
        ),
    )
    if split is not None:
        df = df.with_columns(pl.lit(split).alias("split"))
    return df


def add_split(df: pl.DataFrame, splits: SplitIndices) -> pl.DataFrame:
    return df.with_columns(
        split=pl.when(pl.col("id").is_in(splits.finetune))
        .then(pl.lit("train"))
        .when(pl.col("id").is_in(splits.validation))
        .then(pl.lit("val"))
        .when(pl.col("id").is_in(splits.test))
        .then(pl.lit("test"))
        .otherwise(pl.lit(None))
    )


def impute_unseen_experiments(
    df: pl.DataFrame,
    train_experiments: set[str],
    output_path: Path | None = None,
    match_columns: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Impute experiment_labels that don't appear in the training set.

    For each non-train split, replaces unseen experiment_labels with the most
    common experiment_label from training that matches on the specified columns.

    Args:
        df: DataFrame with experiment_label, cell_type, image_type, split columns
        train_experiments: Set of experiment_labels that appear in training
        output_path: Optional path to save the imputation log as parquet
        match_columns: Columns to match on when finding best experiment.
            Defaults to ["image_type"]. Can be ["image_type"] or
            ["cell_type", "image_type"].

    Returns:
        Tuple of (imputed DataFrame, imputation log DataFrame)
    """
    if match_columns is None:
        match_columns = ["image_type"]

    logger.info(f"Imputing experiments by matching on: {match_columns}")

    # Get all unique splits (excluding train)
    non_train_splits = (
        df.filter(pl.col("split") != "train")
        .select("split")
        .unique()
        .to_series()
        .to_list()
    )

    # Build lookup table: for each combination of match_columns in training,
    # find the most common experiment_label
    train_df = df.filter(pl.col("split") == "train")
    experiment_counts = (
        train_df.group_by([*match_columns, "experiment_label"])
        .agg(pl.len().alias("count"))
        .sort(by="count", descending=True)
    )

    # For each combination of match_columns, get the experiment with highest count
    best_experiment_per_group = (
        experiment_counts.group_by(match_columns)
        .first()
        .select([*match_columns, "experiment_label"])
        .rename({"experiment_label": "best_train_experiment"})
    )

    # Find rows that need imputation (not in train and experiment not seen)
    needs_imputation = df.filter(pl.col("split") != "train").filter(
        ~pl.col("experiment_label").is_in(train_experiments)
    )

    if needs_imputation.height == 0:
        logger.info("No experiment_labels need imputation")
        empty_log = pl.DataFrame(
            {
                "plate_order_read_id": [],
                "split": [],
                "cell_type": [],
                "image_type": [],
                "original_experiment_label": [],
                "imputed_experiment_label": [],
            }
        )
        return df, empty_log

    # Log unique unseen experiments
    unseen_experiments = (
        needs_imputation.select("experiment_label").unique().to_series().to_list()
    )
    logger.info(
        f"Found {len(unseen_experiments)} unseen experiment_labels: {unseen_experiments}"
    )

    # Join with best experiment lookup
    imputation_mapping = (
        needs_imputation.select(
            [
                "plate_order_read_id",
                "split",
                "cell_type",
                "image_type",
                "experiment_label",
            ]
        )
        .join(best_experiment_per_group, on=match_columns, how="left")
        .rename(
            {
                "experiment_label": "original_experiment_label",
                "best_train_experiment": "imputed_experiment_label",
            }
        )
    )

    # Check for any failed imputations (no matching values in training)
    failed_imputations = imputation_mapping.filter(
        pl.col("imputed_experiment_label").is_null()
    )
    if failed_imputations.height > 0:
        logger.warning(
            f"Could not impute {failed_imputations.height} rows - "
            f"no matching {match_columns} in training data"
        )
        # For failed imputations, use the most common experiment overall
        fallback_experiment = (
            experiment_counts.sort(by="count", descending=True)
            .select("experiment_label")
            .head(1)
            .item()
        )
        imputation_mapping = imputation_mapping.with_columns(
            pl.when(pl.col("imputed_experiment_label").is_null())
            .then(pl.lit(fallback_experiment))
            .otherwise(pl.col("imputed_experiment_label"))
            .alias("imputed_experiment_label")
        )
        logger.info(
            f"Using fallback experiment '{fallback_experiment}' for unmatched rows"
        )

    # Create the imputation log
    imputation_log = imputation_mapping.select(
        [
            "plate_order_read_id",
            "split",
            "cell_type",
            "image_type",
            "original_experiment_label",
            "imputed_experiment_label",
        ]
    )

    logger.info(
        f"Imputing {imputation_log.height} rows across {len(non_train_splits)} splits"
    )

    # Log summary per split
    for split in non_train_splits:
        split_log = imputation_log.filter(pl.col("split") == split)
        if split_log.height > 0:
            logger.info(f"  Split '{split}': {split_log.height} imputations")

    # Save imputation log if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imputation_log.write_parquet(output_path)
        logger.info(f"Saved imputation log to {output_path}")

    # Apply imputations to the dataframe
    # Create a mapping from plate_order_read_id to imputed_experiment_label
    imputation_dict = dict(
        zip(
            imputation_mapping["plate_order_read_id"].to_list(),
            imputation_mapping["imputed_experiment_label"].to_list(),
        )
    )

    # Update the dataframe with imputed values
    imputed_df = df.with_columns(
        pl.when(pl.col("plate_order_read_id").is_in(list(imputation_dict.keys())))
        .then(
            pl.col("plate_order_read_id").replace_strict(
                imputation_dict, default=pl.col("experiment_label")
            )
        )
        .otherwise(pl.col("experiment_label"))
        .alias("experiment_label")
    )

    return imputed_df, imputation_log


def create_train_set():
    cross_cell_line_df_train = add_columns(
        CROSS_CELL_LINE_OBS_PATH,
        image_type="brightfield_3channel",
        source="celltype_diversity",
        image_shape=(2048, 2048, 3),
        split="train",
    )
    vcb_df_train = add_split(
        add_columns(
            VCB_OBS_PATH,
            image_type="cellpaint",
            source="vcb",
            image_shape=(2048, 2048, 6),
            split=None,
        ).with_row_index("id"),
        vcb_splits,
    )
    return pl.concat(
        [cross_cell_line_df_train[COMMON_COLUMNS], vcb_df_train[COMMON_COLUMNS]]
    )


def create_inference_set(
    impute_experiments: bool = True,
    match_columns: list[str] | None = None,
):
    """
    Create inference dataset with optional experiment label imputation.

    Args:
        impute_experiments: If True, impute unseen experiment_labels using
            the most common experiment from training with matching columns.
        match_columns: Columns to match on when finding best experiment for
            imputation. Defaults to ["image_type"]. Can be ["image_type"] or
            ["cell_type", "image_type"].
    """
    dart_df = add_columns(
        DART_OBS_PATH,
        image_type="cellpaint",
        source="dart",
        image_shape=(2048, 2048, 6),
        split="test",
    )
    cross_cell_line_df_test = add_columns(
        TEST_OBS_PATH,
        image_type="brightfield_3channel",
        source="celltype_diversity",
        image_shape=(2048, 2048, 3),
        split="test",
    )
    vcb_df = add_columns(
        VCB_OBS_PATH,
        image_type="cellpaint",
        source="vcb",
        image_shape=(2048, 2048, 6),
        split=None,
    ).with_row_index("id")
    vcb_df = add_split(
        vcb_df,
        vcb_splits,
    ).filter(pl.col("split") == "test")
    selected_columns = COMMON_COLUMNS
    combined = (
        pl.concat(
            [
                dart_df[selected_columns],
                cross_cell_line_df_test[selected_columns],
                vcb_df[selected_columns],
            ]
        )
        .with_row_index("zarr_index_generated_raw_counts")
        .with_columns(
            pl.concat_str(["plate_order_read_id", "well_address"], separator=":").alias(
                "obs_id"
            )
        )
    )

    # Optionally impute unseen experiment_labels
    if impute_experiments:
        # Load training set to get known experiments
        train_df = create_train_set()
        train_experiments = set(
            train_df.select("experiment_label").unique().to_series().to_list()
        )
        logger.info(
            f"Found {len(train_experiments)} unique experiment_labels in training set"
        )

        # Combine training and inference for imputation
        # (we need training data to compute the best experiment per cell_type/image_type)
        combined_with_train = pl.concat(
            [
                train_df.select(
                    [
                        "cell_type",
                        "image_type",
                        "experiment_label",
                        "plate_order_read_id",
                        "split",
                    ]
                ),
                combined.select(
                    [
                        "cell_type",
                        "image_type",
                        "experiment_label",
                        "plate_order_read_id",
                        "split",
                    ]
                ),
            ]
        )

        imputation_log_path = Path(
            "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/experiment_imputation_log.parquet"
        )

        _, imputation_log = impute_unseen_experiments(
            combined_with_train,
            train_experiments,
            output_path=imputation_log_path,
            match_columns=match_columns,
        )

        # Apply imputations to the combined inference set
        if imputation_log.height > 0:
            imputation_dict = dict(
                zip(
                    imputation_log["plate_order_read_id"].to_list(),
                    imputation_log["imputed_experiment_label"].to_list(),
                )
            )
            combined = combined.with_columns(
                pl.when(
                    pl.col("plate_order_read_id").is_in(list(imputation_dict.keys()))
                )
                .then(
                    pl.col("plate_order_read_id").replace_strict(
                        imputation_dict, default=pl.col("experiment_label")
                    )
                )
                .otherwise(pl.col("experiment_label"))
                .alias("experiment_label")
            )
            logger.info(f"Applied {len(imputation_dict)} experiment_label imputations")

    combined.write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/inference_set.parquet"
    )
    combined.filter(pl.col("source") == "dart").write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/dart_inference_set.parquet"
    )
    combined.filter(pl.col("source") == "vcb").write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/vcb_inference_set.parquet"
    )
    combined.filter(pl.col("source") == "celltype_diversity").write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/celltype_diversity_inference_set.parquet"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # match_columns=["image_type"] matches only on image_type (default)
    # match_columns=["cell_type", "image_type"] matches on both
    create_inference_set(impute_experiments=True, match_columns=["image_type"])
