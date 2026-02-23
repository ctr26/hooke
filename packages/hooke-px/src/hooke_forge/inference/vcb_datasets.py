"""VCB dataset configurations and transformations.

Defines known VCB dataset types and provides transformation functions
to convert VCB observation parquet files to the inference format.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

# Base path for VCB datasets
VCB_BASE_PATH = "/rxrx/data/valence/internal_benchmarking/vcds1"

# Known VCB dataset configurations
VCB_DATASETS: dict[str, dict[str, Any]] = {
    "drugscreen": {
        "image_type": "cellpaint",
        "shape": [2048, 2048, 6],
        "source": "drugscreen",
        "obs_path": f"{VCB_BASE_PATH}/drugscreen__cell_paint__v1_2/drugscreen__cell_paint__v1_2_obs.parquet",
    },
    "cross_cell_line": {
        "image_type": "brightfield_3channel",
        "shape": [2048, 2048, 3],
        "source": "cross_cell_line",
        "obs_path": f"{VCB_BASE_PATH}/cross_cell_line__brightfield__v1_1/cross_cell_line__brightfield__v1_1_obs.parquet",
    },
}


def transform_vcb_dataset(df: pl.DataFrame, dataset_type: str) -> pl.DataFrame:
    """Transform VCB obs parquet to inference format.

    Adds required columns for inference that are not present in raw VCB
    observation files:
    - image_type: Type of microscopy image (cellpaint, brightfield_3channel)
    - shape: Image dimensions [H, W, C]
    - source: Dataset source identifier
    - rec_id: Extracted from perturbations.source_id
    - concentration: Extracted from perturbations.concentration

    Args:
        df: Raw VCB observation DataFrame with perturbations column
        dataset_type: VCB dataset type ("drugscreen" or "cross_cell_line")

    Returns:
        Transformed DataFrame with inference-required columns

    Raises:
        ValueError: If dataset_type is not a known VCB dataset
    """
    if dataset_type not in VCB_DATASETS:
        available = list(VCB_DATASETS.keys())
        raise ValueError(
            f"Unknown VCB dataset type: {dataset_type}. Available types: {available}"
        )

    config = VCB_DATASETS[dataset_type]
    log.info(f"Transforming VCB dataset: {dataset_type}")
    log.info(f"  image_type: {config['image_type']}")
    log.info(f"  shape: {config['shape']}")

    # Check for perturbations column
    if "perturbations" not in df.columns:
        raise ValueError(
            "VCB dataset must have 'perturbations' column. "
            "This doesn't appear to be a VCB-formatted observation file."
        )

    return df.with_columns(
        pl.lit(config["image_type"]).alias("image_type"),
        pl.lit(config["shape"]).alias("shape"),
        pl.lit(config["source"]).alias("source"),
        rec_id=pl.col("perturbations").list.eval(
            pl.element().struct.field("source_id")
        ),
        concentration=pl.col("perturbations").list.eval(
            pl.element().struct.field("concentration").cast(pl.Float64).cast(pl.String)
        ),
    )


def get_vcb_dataset_types() -> list[str]:
    """Get list of available VCB dataset types."""
    return list(VCB_DATASETS.keys())


def get_vcb_obs_path(dataset_type: str) -> str:
    """Get the default obs parquet path for a VCB dataset type.

    Args:
        dataset_type: VCB dataset type ("drugscreen" or "cross_cell_line")

    Returns:
        Path to the obs parquet file

    Raises:
        ValueError: If dataset_type is not a known VCB dataset
    """
    if dataset_type not in VCB_DATASETS:
        available = list(VCB_DATASETS.keys())
        raise ValueError(
            f"Unknown VCB dataset type: {dataset_type}. Available types: {available}"
        )
    return VCB_DATASETS[dataset_type]["obs_path"]
