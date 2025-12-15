#!/usr/bin/env python3
"""Prepare HuggingFace Dataset cache from parquet metadata.

This script converts a parquet metadata file to a HuggingFace Dataset format
with pre-tokenized columns for O(1) random access during training.

Uses chunked processing to handle large datasets without OOM.

Usage:
    python scripts/prepare_hf_cache.py --path /path/to/metadata.parquet
    python scripts/prepare_hf_cache.py --path /path/to/metadata.parquet --splits train val test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import polars as pl
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from adaptor import DataFrameTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def tokenize_chunk(
    chunk_df: pl.DataFrame,
    tokenizer: DataFrameTokenizer,
    pad_length: int,
) -> dict:
    """Tokenize a chunk of rows directly from Polars DataFrame.

    This is more memory efficient than going through HF Dataset.map().
    """
    n = len(chunk_df)

    # Pre-allocate lists
    image_paths = chunk_df["image_path"].to_list()
    rec_id_raw = chunk_df["rec_id"].to_list()
    conc_raw = chunk_df["concentration"].to_list()
    cell_type_raw = chunk_df["cell_type"].to_list()
    image_type_raw = chunk_df["image_type"].to_list()
    experiment_raw = chunk_df["experiment_label"].to_list()
    well_address_raw = chunk_df["well_address"].to_list()

    rec_ids = []
    concentrations = []
    rec_id_lens = []
    cell_types = []
    image_types = []
    experiment_labels = []
    well_addresses = []

    for i in range(n):
        # Tokenize list columns
        rec_id = tokenizer.rec_id_tokenizer(rec_id_raw[i])
        conc = tokenizer.concentration_tokenizer(conc_raw[i])

        # Pad to fixed length
        rec_id_len = len(rec_id)
        rec_id_padded = rec_id + [0] * (pad_length - rec_id_len)
        conc_padded = conc + [0] * (pad_length - len(conc))

        rec_ids.append(rec_id_padded)
        concentrations.append(conc_padded)
        rec_id_lens.append(rec_id_len)

        # Tokenize scalar columns
        cell_types.append(tokenizer.cell_type_tokenizer(cell_type_raw[i])[0])
        image_types.append(tokenizer.image_type_tokenizer(image_type_raw[i])[0])
        experiment_labels.append(tokenizer.experiment_tokenizer(experiment_raw[i])[0])
        well_addresses.append(tokenizer.well_address_tokenizer(well_address_raw[i])[0])

    return {
        "image_path": image_paths,
        "rec_id": rec_ids,
        "concentration": concentrations,
        "rec_id_len": rec_id_lens,
        "cell_type": cell_types,
        "image_type": image_types,
        "experiment_label": experiment_labels,
        "well_address": well_addresses,
    }


def prepare_cache(
    parquet_path: str,
    splits: list[str] | None = None,
    pad_length: int = 8,
    chunk_size: int = 1_000_000,
) -> Path:
    """Prepare HuggingFace Dataset cache from parquet metadata.

    Args:
        parquet_path: Path to source parquet file
        splits: List of splits to process (default: ["train", "val", "test"])
        pad_length: Padding length for list columns
        chunk_size: Number of rows to process at a time (for memory efficiency)

    Returns:
        Path to cache directory
    """
    if splits is None:
        splits = ["train", "val", "test"]

    overall_start = time.perf_counter()

    parquet_path = Path(parquet_path)
    cache_dir = parquet_path.with_suffix(".cache")
    cache_dir.mkdir(exist_ok=True)

    log.info(f"{'=' * 60}")
    log.info("Preparing HuggingFace Dataset Cache")
    log.info(f"{'=' * 60}")
    log.info(f"Source: {parquet_path}")
    log.info(f"Cache:  {cache_dir}")
    log.info(f"Splits: {splits}")
    log.info(f"Pad length: {pad_length}")
    log.info(f"Chunk size: {chunk_size:,}")
    log.info(f"{'=' * 60}")

    log.info(f"\nLoading parquet: {parquet_path}")
    start = time.perf_counter()
    df = pl.read_parquet(parquet_path)
    log.info(f"Loaded {len(df):,} rows in {time.perf_counter() - start:.1f}s")

    # Build tokenizer from full data (needs all unique values for consistent encoding)
    log.info("Building tokenizer from full dataset...")
    start = time.perf_counter()
    tokenizer = DataFrameTokenizer(df, pad_length=pad_length)
    log.info(f"Tokenizer built in {time.perf_counter() - start:.1f}s")
    log.info(f"  rec_id vocab size: {len(tokenizer.rec_id_tokenizer):,}")
    log.info(f"  concentration vocab size: {len(tokenizer.concentration_tokenizer):,}")
    log.info(f"  cell_type vocab size: {len(tokenizer.cell_type_tokenizer):,}")
    log.info(f"  experiment vocab size: {len(tokenizer.experiment_tokenizer):,}")
    log.info(f"  image_type vocab size: {len(tokenizer.image_type_tokenizer):,}")
    log.info(f"  well_address vocab size: {len(tokenizer.well_address_tokenizer):,}")

    # Save tokenizer
    tokenizer_path = cache_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer.state_dict(), f)
    log.info(f"Saved tokenizer to {tokenizer_path}")

    # Define features for efficient storage
    features = Features(
        {
            "image_path": Value("string"),
            "rec_id": Sequence(Value("int32"), length=pad_length),
            "concentration": Sequence(Value("int32"), length=pad_length),
            "rec_id_len": Value("int8"),
            "cell_type": Value("int32"),
            "image_type": Value("int32"),
            "experiment_label": Value("int32"),
            "well_address": Value("int32"),
        }
    )

    # Process each split
    for split_idx, split in enumerate(splits):
        log.info(f"\n{'=' * 60}")
        log.info(f"Processing split {split_idx + 1}/{len(splits)}: {split}")
        log.info(f"{'=' * 60}")
        split_dir = cache_dir / split

        # Filter to split
        log.info("  Filtering DataFrame...")
        filter_start = time.perf_counter()
        split_df = df.filter(pl.col("split") == split)
        n_rows = len(split_df)
        log.info(
            f"  Filtered to {n_rows:,} rows in {time.perf_counter() - filter_start:.1f}s"
        )

        if n_rows == 0:
            log.warning(f"  No rows found for split '{split}', skipping")
            continue

        # Process in chunks to avoid OOM
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        log.info(f"  Processing in {n_chunks} chunks of up to {chunk_size:,} rows...")

        chunk_datasets = []
        chunk_start_time = time.perf_counter()

        for chunk_idx in tqdm(range(n_chunks), desc=f"  {split} chunks"):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, n_rows)

            # Slice the chunk
            chunk_df = split_df.slice(start_row, end_row - start_row)

            # Tokenize directly
            tokenized = tokenize_chunk(chunk_df, tokenizer, pad_length)

            # Create dataset from tokenized data
            chunk_ds = Dataset.from_dict(tokenized, features=features)
            chunk_datasets.append(chunk_ds)

            # Log progress periodically
            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                elapsed = time.perf_counter() - chunk_start_time
                rows_done = end_row
                rate = rows_done / elapsed
                eta = (n_rows - rows_done) / rate if rate > 0 else 0
                log.info(
                    f"    Chunk {chunk_idx + 1}/{n_chunks}: "
                    f"{rows_done:,}/{n_rows:,} rows "
                    f"({rate:,.0f} rows/s, ETA: {eta / 60:.1f}m)"
                )

        log.info(
            f"  All chunks processed in {time.perf_counter() - chunk_start_time:.1f}s"
        )

        # Concatenate all chunks
        log.info(f"  Concatenating {len(chunk_datasets)} chunks...")
        concat_start = time.perf_counter()
        hf_dataset = concatenate_datasets(chunk_datasets)
        log.info(f"  Concatenated in {time.perf_counter() - concat_start:.1f}s")

        # Clear chunk datasets to free memory
        del chunk_datasets

        # Save to disk
        log.info(f"  Saving to {split_dir}...")
        start = time.perf_counter()
        hf_dataset.save_to_disk(split_dir)
        save_time = time.perf_counter() - start

        # Log split size
        split_size = sum(f.stat().st_size for f in split_dir.rglob("*") if f.is_file())
        log.info(f"  Saved in {save_time:.1f}s ({split_size / 1e9:.2f} GB)")

        # Clear dataset to free memory before next split
        del hf_dataset

    total_elapsed = time.perf_counter() - overall_start

    log.info(f"\n{'=' * 60}")
    log.info("COMPLETE")
    log.info(f"{'=' * 60}")
    log.info(f"Cache prepared at: {cache_dir}")

    # Log cache size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    log.info(f"Total cache size: {total_size / 1e9:.2f} GB")
    log.info(f"Total time: {total_elapsed / 60:.1f} minutes ({total_elapsed:.0f}s)")

    return cache_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace Dataset cache from parquet metadata"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to source parquet file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--pad-length",
        type=int,
        default=8,
        help="Padding length for list columns (default: 8)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Rows per chunk for memory-efficient processing (default: 1000000)",
    )
    args = parser.parse_args()

    prepare_cache(
        parquet_path=args.path,
        splits=args.splits,
        pad_length=args.pad_length,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
