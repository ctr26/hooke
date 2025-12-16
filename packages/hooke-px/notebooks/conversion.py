from pathlib import Path
import polars as pl
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, Any

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
    "/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_observation_random__v1.json"
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


def create_train_set():
    cross_cell_line_df_train = add_columns(
        CROSS_CELL_LINE_OBS_PATH,
        image_type="brightfield_3channel",
        source="celltype_diversity",
        image_shape=[2048, 2048, 3],
        split="train",
    )
    vcb_df_train = add_split(
        add_columns(
            VCB_OBS_PATH,
            image_type="cellpaint",
            source="vcb",
            image_shape=[2048, 2048, 6],
            split=None,
        ),
        vcb_splits,
    )
    return pl.concat([cross_cell_line_df_train, vcb_df_train])


def create_inference_set():
    dart_df = add_columns(
        DART_OBS_PATH,
        image_type="cellpaint",
        source="dart",
        image_shape=[2048, 2048, 6],
        split="test",
    )
    cross_cell_line_df_test = add_columns(
        TEST_OBS_PATH,
        image_type="brightfield_3channel",
        source="celltype_diversity",
        image_shape=[2048, 2048, 3],
        split="test",
    )
    vcb_df = add_split(
        add_columns(
            VCB_OBS_PATH,
            image_type="cellpaint",
            source="vcb",
            image_shape=[2048, 2048, 6],
            split=None,
        ).with_row_index("id"),
        vcb_splits,
    ).filter(pl.col("split") == "test")
    selected_columns = [
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
    combined = (
        pl.concat(
            [
                dart_df[selected_columns],
                cross_cell_line_df_test[selected_columns],
                # vcb_df[selected_columns],
            ]
        )
        .with_row_index("zarr_index_generated_raw_counts")
        .with_columns(
            pl.concat_str(["plate_order_read_id", "well_address"], separator=":").alias(
                "obs_id"
            )
        )
    )

    combined.write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/inference_set.parquet"
    )
    combined.filter(pl.col("source") == "dart").write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/dart_inference_set.parquet"
    )
    # combined.filter(pl.col("source") == "vcb").write_parquet(
    #    "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/vcb_inference_set.parquet"
    # )
    combined.filter(pl.col("source") == "celltype_diversity").write_parquet(
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/celltype_diversity_inference_set.parquet"
    )


if __name__ == "__main__":
    create_inference_set()
