"""Standalone gene selection preprocessing utility for Tx data.

This module provides HVG (Highly Variable Gene) selection functionality
that was previously embedded in TxDataset. It allows preprocessing gene
subsets independently before training to enable better workflow separation
and reusability.

Usage:
    # Console script
    hooke-gene-select --gene_selection.metadata_path /path/to/metadata.parquet \
                      --gene_selection.select_strategy hvg \
                      --gene_selection.n_features 5000

    # Python API
    gene_subset_path = create_gene_subset(
        metadata_path="/path/to/metadata.parquet",
        zarr_path="/path/to/features.zarr",
        var_metadata_path="/path/to/var.parquet",
        train_split="train",
        select_strategy="hvg",
        n_features=5000
    )
"""

import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import ornamentalist
import polars as pl
import scanpy as sc
import zarr
from scipy import sparse

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


@dataclasses.dataclass
class GeneSelectionConfig:
    """Configuration for gene selection preprocessing."""

    select_strategy: Literal["hvg", "top", "all"] = "hvg"
    n_features: int = 5000
    target_sum: int = 4000  # Normalization target
    exclude_ercc: bool = True


def _get_gene_selection_cache_key(
    config: GeneSelectionConfig,
    metadata_path: str | Path,
    zarr_path: str | Path,
    var_metadata_path: str | Path,
    train_split: str,
) -> str:
    """Generate a cache key for gene selection based on config and data fingerprints."""
    # Include gene selection config parameters
    config_dict = dataclasses.asdict(config)

    # Get file modification times as fingerprints
    metadata_mtime = Path(metadata_path).stat().st_mtime
    zarr_mtime = Path(zarr_path).stat().st_mtime
    var_mtime = Path(var_metadata_path).stat().st_mtime

    # Create composite key
    key_data = {
        "config": config_dict,
        "metadata_mtime": metadata_mtime,
        "zarr_mtime": zarr_mtime,
        "var_mtime": var_mtime,
        "metadata_path": str(metadata_path),
        "zarr_path": str(zarr_path),
        "var_path": str(var_metadata_path),
        "train_split": train_split,
    }

    # Hash to create a stable key
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]  # Use first 16 chars


def _get_gene_selection_cache_path(cache_key: str) -> Path:
    """Get the cache file path for given cache key."""
    cache_dir = Path.home() / ".cache" / "hooke_forge" / "gene_subsets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gene_subset_{cache_key}.npz"


def _save_gene_selection_cache(
    cache_key: str,
    gene_mask: np.ndarray,
    hvg_indices: np.ndarray,
    gene_symbols: np.ndarray,
    metadata: dict,
) -> None:
    """Save gene selection results to cache."""
    cache_path = _get_gene_selection_cache_path(cache_key)
    try:
        np.savez_compressed(
            cache_path,
            gene_mask=gene_mask,
            hvg_indices=hvg_indices,
            gene_symbols=gene_symbols,
            metadata=np.array([metadata]),  # Wrap dict in array for npz
        )
        _log.info(f"Saved gene selection cache to {cache_path}")
    except Exception as e:
        _log.warning(f"Failed to save gene selection cache: {e}")


def _load_gene_selection_cache(cache_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    """Load gene selection results from cache. Returns (gene_mask, hvg_indices, gene_symbols, metadata) or None."""
    cache_path = _get_gene_selection_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        gene_mask = data["gene_mask"]
        hvg_indices = data["hvg_indices"]
        gene_symbols = data["gene_symbols"]
        metadata = data["metadata"].item()  # Extract dict from array wrapper
        _log.info(f"Loaded gene selection cache from {cache_path}")
        return gene_mask, hvg_indices, gene_symbols, metadata
    except Exception as e:
        _log.warning(f"Failed to load gene selection cache: {e}")
        # Remove corrupted cache file
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def _get_highly_variable_genes(
    adata: sc.AnnData,
    n_features: int,
    target_sum: int,
) -> np.ndarray:
    """HVG selection using scanpy (matches original implementation)."""
    # Normalize and log transform
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=target_sum)
    sc.pp.log1p(adata_copy)

    # Compute highly variable genes (don't subset, just mark them)
    sc.pp.highly_variable_genes(
        adata_copy,
        n_top_genes=n_features,
        subset=False,  # Keep all genes, just mark HVGs
    )

    # Get indices of highly variable genes
    hvg_mask = adata_copy.var["highly_variable"].values
    hvg_indices = np.where(hvg_mask)[0]

    # Ensure we get exactly the requested number of features
    if len(hvg_indices) != n_features:
        _log.warning(f"Expected {n_features} HVGs, got {len(hvg_indices)}")
        # Take the first n_features to be safe
        hvg_indices = hvg_indices[:n_features]

    return hvg_indices


def _get_top_genes(
    adata: sc.AnnData,
    n_features: int,
    target_sum: int,
) -> np.ndarray:
    """Top genes by mean expression (matches original implementation)."""
    # Normalize and log transform
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=target_sum)
    sc.pp.log1p(adata_copy)

    # Sort by mean expression
    if sparse.issparse(adata_copy.X):
        mean_expr = np.array(adata_copy.X.mean(axis=0)).squeeze()
    else:
        mean_expr = adata_copy.X.mean(axis=0)

    top_indices = np.argsort(-mean_expr)[:n_features]
    return top_indices


@ornamentalist.configure(name="gene_selection")
def create_gene_subset(
    *,
    metadata_path: str = ornamentalist.Configurable[
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/training_trek_v1_0_obs.parquet"
    ],
    zarr_path: str = ornamentalist.Configurable[
        "/rxrx/data/user/ali.denton/tmp/training_trek__v1_0/training_trek__v1_0_features.zarr"
    ],
    var_metadata_path: str = ornamentalist.Configurable[
        "/rxrx/data/user/ali.denton/tmp/training_trek__v1_0/training_trek__v1_0_var.parquet"
    ],
    train_split: str = ornamentalist.Configurable["train"],  # Split name to use for HVG computation
    output_path: str = ornamentalist.Configurable[""],  # Auto-generated if empty
    select_strategy: Literal["hvg", "top", "all"] = ornamentalist.Configurable["hvg"],
    n_features: int = ornamentalist.Configurable[5000],
    target_sum: int = ornamentalist.Configurable[4000],
    exclude_ercc: bool = ornamentalist.Configurable[True],
    force_recompute: bool = ornamentalist.Configurable[False],
) -> str:
    """Create gene subset file for use with TxDataset.

    Computes HVGs only on training split to avoid data leakage.

    Args:
        metadata_path: Path to observation metadata parquet file
        zarr_path: Path to zarr features file
        var_metadata_path: Path to variable metadata parquet file
        train_split: Split name to use for HVG computation (e.g., "train")
        output_path: Output path for gene subset .npz file (auto-generated if empty)
        select_strategy: Gene selection strategy ("hvg", "top", or "all")
        n_features: Number of features to select
        target_sum: Normalization target for scanpy preprocessing
        exclude_ercc: Whether to exclude ERCC spike-in controls
        force_recompute: Force recomputation even if cache exists

    Returns:
        Path to created gene subset .npz file
    """
    # Create config object
    config = GeneSelectionConfig(
        select_strategy=select_strategy,
        n_features=n_features,
        target_sum=target_sum,
        exclude_ercc=exclude_ercc,
    )

    # Generate cache key and check for cached results
    cache_key = _get_gene_selection_cache_key(config, metadata_path, zarr_path, var_metadata_path, train_split)

    # Generate output path if not provided
    if not output_path:
        timestamp = int(time.time())
        output_path = f"gene_subset_{select_strategy}_{n_features}_{timestamp}.npz"
    output_path = Path(output_path).resolve()

    # Try to load from cache first
    if not force_recompute:
        cached_results = _load_gene_selection_cache(cache_key)
        if cached_results is not None:
            gene_mask, hvg_indices, gene_symbols, metadata = cached_results
            _log.info(f"Using cached gene selection with {len(hvg_indices)} features")

            # Validate cached results have correct dimensions
            if len(hvg_indices) != n_features:
                _log.warning(f"Cached selection has {len(hvg_indices)} features, expected {n_features}. Recomputing...")
                # Clear invalid cache entry
                cache_path = _get_gene_selection_cache_path(cache_key)
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            else:
                # Save to output path and return
                np.savez_compressed(
                    output_path,
                    gene_mask=gene_mask,
                    hvg_indices=hvg_indices,
                    gene_symbols=gene_symbols,
                    metadata=np.array([metadata]),
                )
                _log.info(f"Saved gene subset to {output_path}")
                return str(output_path)

    # Cache miss or forced recompute - compute gene selection
    _log.info("Computing gene selection (this may take a while)...")

    # Load observation metadata
    df = pl.read_parquet(metadata_path)
    _log.info(f"Loaded metadata with {len(df)} observations")

    # Filter to training split only (avoid data leakage)
    train_df = df.filter(pl.col("split") == train_split)
    if len(train_df) == 0:
        raise ValueError(f"No samples found for train_split='{train_split}'")
    _log.info(f"Using {len(train_df)} training samples for gene selection")

    # Load variable metadata
    var_df = pl.read_parquet(var_metadata_path)
    _log.info(f"Loaded variable metadata with {len(var_df)} genes")

    # Get zarr row indices for training samples
    if "zarr_row_idx" not in train_df.columns:
        train_df = train_df.with_row_index("zarr_row_idx")
    train_zarr_indices = train_df["zarr_row_idx"].to_numpy()

    # Open zarr file and validate dimensions
    zarr_file = zarr.open(zarr_path)
    _log.info(f"Opened zarr file with shape {zarr_file.shape}")

    # Sample subset for HVG computation (from training data only)
    # Use fixed seed for reproducibility within the same cache key
    np.random.seed(hash(cache_key) % (2**32))
    sample_size = min(50_000, len(train_zarr_indices))
    sample_indices = np.random.choice(train_zarr_indices, sample_size, replace=False)
    _log.info(f"Using {sample_size} samples for HVG computation")

    # Load expression data for training samples only
    X_sample = np.array([zarr_file[int(i)] for i in sample_indices])
    adata_sample = sc.AnnData(X=X_sample)
    adata_sample.var_names = var_df["gene_symbol"].fill_null("unknown")

    # Filter ERCC controls if requested
    if exclude_ercc:
        # Handle nulls explicitly and use starts_with
        ercc_mask = var_df["source_gene_symbol"].str.starts_with("ERCC-").fill_null(False)
        gene_mask = ~ercc_mask
        gene_mask_np = gene_mask.to_numpy()
        adata_sample = adata_sample[:, gene_mask_np]
        _log.info(f"Excluded ERCC controls, {gene_mask_np.sum()} genes remaining")
    else:
        gene_mask_np = np.ones(len(var_df), dtype=bool)

    # Perform gene selection based on strategy
    if select_strategy == "hvg":
        hvg_indices = _get_highly_variable_genes(adata_sample, n_features, target_sum)
    elif select_strategy == "top":
        hvg_indices = _get_top_genes(adata_sample, n_features, target_sum)
    elif select_strategy == "all":
        # For "all" strategy, still respect n_features limit
        all_indices = np.arange(adata_sample.n_vars)
        if len(all_indices) > n_features:
            _log.warning(f"'all' strategy has {len(all_indices)} features, limiting to {n_features}")
            hvg_indices = all_indices[:n_features]
        else:
            hvg_indices = all_indices
    else:
        raise ValueError(f"Unsupported select_strategy: {select_strategy}")

    # Final safety check: ensure we have exactly the expected number of features
    if len(hvg_indices) != n_features:
        _log.error(f"Gene selection returned {len(hvg_indices)} features, expected {n_features}")
        # Truncate or pad as needed
        if len(hvg_indices) > n_features:
            hvg_indices = hvg_indices[:n_features]
            _log.warning(f"Truncated to {n_features} features")
        else:
            raise ValueError(f"Gene selection returned insufficient features: {len(hvg_indices)} < {n_features}")

    # Get gene symbols for selected features (after filtering)
    if exclude_ercc:
        filtered_var_df = var_df.filter(~ercc_mask)
        gene_symbols = filtered_var_df["gene_symbol"].fill_null("unknown")[hvg_indices].to_numpy()
    else:
        gene_symbols = var_df["gene_symbol"].fill_null("unknown")[hvg_indices].to_numpy()

    # Create metadata for output file
    metadata = {
        "config": dataclasses.asdict(config),
        "metadata_path": str(metadata_path),
        "zarr_path": str(zarr_path),
        "var_metadata_path": str(var_metadata_path),
        "train_split": train_split,
        "created_at": time.time(),
        "cache_key": cache_key,
        "n_train_samples": len(train_df),
        "n_sample_for_hvg": sample_size,
    }

    # Save results to cache
    _save_gene_selection_cache(cache_key, gene_mask_np, hvg_indices, gene_symbols, metadata)

    # Save to output file
    np.savez_compressed(
        output_path,
        gene_mask=gene_mask_np,
        hvg_indices=hvg_indices,
        gene_symbols=gene_symbols,
        metadata=np.array([metadata]),  # Wrap dict in array for npz
    )

    _log.info(f"Created gene subset with {len(hvg_indices)} features")
    _log.info(f"Saved gene subset to {output_path}")
    return str(output_path)


def cli():
    """Console script entry point: hooke-gene-select"""
    configs = ornamentalist.cli()
    for config in configs:
        ornamentalist.setup(config, force=True)
        result_path = create_gene_subset(**config["gene_selection"])
        print(f"Gene subset created at: {result_path}")


if __name__ == "__main__":
    cli()
