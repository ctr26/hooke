"""Worker script for transformer encoder embedding extraction.

Extracts TransformerEncoder embeddings for different conditioning scenarios,
without running the full DiT model or generating images.

Three embedding cases:
- Case 1: (rec_id, concentration, experiment, cell_type) with well_address marginalized
- Case 2: (rec_id, concentration, experiment, cell_type, well_address) for all well addresses
- Case 3: (rec_id, concentration, cell_type) with experiment and well_address marginalized

Usage:
    python run_adaptor_embedding.py --worker_dir /path/to/worker_0 --config /path/to/config.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import ornamentalist
import polars as pl
import torch
import zarr
from torch.utils.data import DataLoader

from context_encoders import DataFrameTokenizer, TransformerEncoder
from main import (
    REC_ID_DIM,
    CONCENTRATION_DIM,
    CELL_TYPE_DIM,
    ASSAY_TYPE_DIM,
    EXPERIMENT_DIM,
    WELL_ADDRESS_DIM,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod prefix from state dict keys (added by torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    return new_state_dict


def load_transformer_encoder(
    checkpoint_path: str,
    device: torch.device,
    hidden_size: int = 1152,
) -> tuple[TransformerEncoder, DataFrameTokenizer]:
    """Load only TransformerEncoder and tokenizer from checkpoint (not full DiT).

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the transformer encoder on.
        hidden_size: Hidden size of the model (default 1152 for DiT-XL/2).

    Returns:
        Tuple of (transformer_encoder, tokenizer)
    """
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Restore tokenizer from checkpoint
    tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    # Build transformer encoder with fixed vocab sizes (must match training dimensions)
    transformer_encoder = TransformerEncoder(
        hidden_size=hidden_size,
        rec_id_dim=REC_ID_DIM,
        concentration_dim=CONCENTRATION_DIM,
        cell_type_dim=CELL_TYPE_DIM,
        experiment_dim=EXPERIMENT_DIM,
        assay_type_dim=ASSAY_TYPE_DIM,
        well_address_dim=WELL_ADDRESS_DIM,
    )

    # Extract transformer encoder weights from EMA state_dict
    ema_state = strip_orig_mod_prefix(state["ema"])

    # The EMA state has keys like "module.transformer_encoder.rec_id_embedder.embedding_table.weight"
    # We need to extract just the transformer_encoder part
    transformer_encoder_state = {}
    prefix = "module.transformer_encoder."
    for key, value in ema_state.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            transformer_encoder_state[new_key] = value

    transformer_encoder.load_state_dict(transformer_encoder_state)
    transformer_encoder.to(device)
    transformer_encoder.eval()

    log.info(
        f"Loaded transformer encoder with {sum(p.numel() for p in transformer_encoder.parameters())} parameters"
    )
    return transformer_encoder, tokenizer


class MetadataDataset(torch.utils.data.Dataset):
    """Lightweight dataset that only yields tokenized metadata (no image loading)."""

    def __init__(
        self,
        metadata: pl.DataFrame,
        tokenizer: DataFrameTokenizer,
        marginal_dims: dict[str, int] | None = None,
    ):
        """
        Args:
            metadata: DataFrame with metadata columns.
            tokenizer: Tokenizer for converting metadata to tokens.
            marginal_dims: Dict mapping field names to their null token values.
                           e.g., {"well_address": 1380, "experiment_label": 4159}
        """
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.marginal_dims = marginal_dims or {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.row(index, named=True)
        tokens = self.tokenizer(row)

        # Override marginalized dimensions with null token
        for field, null_val in self.marginal_dims.items():
            tokens[field] = torch.tensor(null_val, dtype=torch.long)

        result = {"meta": tokens}
        if "zarr_index" in row:
            result["zarr_index"] = row["zarr_index"]
        return result


@torch.inference_mode()
def compute_embeddings(
    transformer_encoder: TransformerEncoder,
    batch_meta: dict,
    device: torch.device,
) -> np.ndarray:
    """Forward pass through transformer_encoder only - returns (B, hidden_size) embeddings."""
    meta = {k: v.to(device) for k, v in batch_meta.items()}
    embedding = transformer_encoder(
        rec_id=meta["rec_id"],
        concentration=meta["concentration"],
        comp_mask=meta["comp_mask"],
        cell_type=meta["cell_type"],
        experiment_label=meta["experiment_label"],
        assay_type=meta["assay_type"],
        well_address=meta["well_address"],
    )
    return embedding.cpu().numpy()


@torch.inference_mode()
def process_case2_batch(
    transformer_encoder: TransformerEncoder,
    batch_meta: dict,
    num_well_addresses: int,
    device: torch.device,
) -> np.ndarray:
    """For each row, compute embeddings for all well_address values.

    Args:
        transformer_encoder: The TransformerEncoder model.
        batch_meta: Batch of tokenized metadata.
        num_well_addresses: Number of well addresses to iterate over.
        device: Device to compute on.

    Returns:
        Embeddings of shape (B, num_well_addresses, hidden_size)
    """
    B = batch_meta["rec_id"].shape[0]

    # Expand batch to B * num_well_addresses
    expanded_meta = {
        k: v.to(device).repeat_interleave(num_well_addresses, dim=0)
        for k, v in batch_meta.items()
    }

    # Create well_address tensor: [0,1,2,...,N-1, 0,1,2,...,N-1, ...]
    well_addresses = torch.arange(num_well_addresses, device=device).repeat(B)
    expanded_meta["well_address"] = well_addresses

    embedding = transformer_encoder(
        rec_id=expanded_meta["rec_id"],
        concentration=expanded_meta["concentration"],
        comp_mask=expanded_meta["comp_mask"],
        cell_type=expanded_meta["cell_type"],
        experiment_label=expanded_meta["experiment_label"],
        assay_type=expanded_meta["assay_type"],
        well_address=expanded_meta["well_address"],
    )

    # Reshape to (B, num_well_addresses, hidden_size)
    hidden_size = embedding.shape[-1]
    return embedding.cpu().numpy().reshape(B, num_well_addresses, hidden_size)


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for MetadataDataset."""
    meta_keys = batch[0]["meta"].keys()
    collated = {
        "meta": {k: torch.stack([b["meta"][k] for b in batch]) for k in meta_keys},
    }
    if "zarr_index" in batch[0]:
        collated["zarr_index"] = np.array([b["zarr_index"] for b in batch])
    return collated


@ornamentalist.configure()
def get_worker_config(
    checkpoint_path: str = ornamentalist.Configurable[""],
    zarr_dir: str = ornamentalist.Configurable[""],
    case: int = ornamentalist.Configurable[1],
    batch_size: int = ornamentalist.Configurable[256],
    num_workers: int = ornamentalist.Configurable[4],
    hidden_size: int = ornamentalist.Configurable[1152],
):
    return (
        checkpoint_path,
        zarr_dir,
        case,
        batch_size,
        num_workers,
        hidden_size,
    )


def run_worker(worker_dir: str, config_path: str):
    """Main worker function."""
    worker_dir = Path(worker_dir)
    chunk_path = worker_dir / "chunk.parquet"

    with open(config_path) as f:
        config = json.load(f)

    ornamentalist.setup(config, force=True)

    # Get injected parameters
    (
        checkpoint_path,
        zarr_dir,
        case,
        batch_size,
        num_workers,
        hidden_size,
    ) = get_worker_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Worker directory: {worker_dir}")
    log.info(f"Case: {case}")

    # Load chunk metadata
    df = pl.read_parquet(chunk_path)
    incomplete_mask = ~df["complete"]
    df_incomplete = df.filter(incomplete_mask)

    if len(df_incomplete) == 0:
        log.info("All rows already complete")
        return

    log.info(f"Processing {len(df_incomplete)} incomplete rows out of {len(df)} total")

    # Load transformer encoder
    log.info(f"Loading checkpoint: {checkpoint_path}")
    transformer_encoder, tokenizer = load_transformer_encoder(checkpoint_path, device, hidden_size)

    # Determine which dimensions to marginalize based on case
    if case == 1:
        # Marginalize well_address only
        marginal_dims = {"well_address": len(tokenizer.well_address_tokenizer)}
    elif case == 2:
        # No marginalization - we'll iterate over well_addresses in process_case2_batch
        marginal_dims = {}
    elif case == 3:
        # Marginalize both experiment and well_address
        marginal_dims = {
            "experiment_label": len(tokenizer.experiment_tokenizer),
            "well_address": len(tokenizer.well_address_tokenizer),
        }
    else:
        raise ValueError(f"Invalid case: {case}")

    log.info(f"Marginal dimensions: {marginal_dims}")

    # Create dataset and loader
    dataset = MetadataDataset(
        metadata=df_incomplete,
        tokenizer=tokenizer,
        marginal_dims=marginal_dims,
    )

    # For case 2, use smaller batch size since we expand by num_well_addresses
    effective_batch_size = batch_size if case != 2 else max(1, batch_size // 64)

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Open shared zarr array
    emb_zarr = zarr.open(f"{zarr_dir}/transformer_encoder_emb.zarr", mode="r+")

    # Process batches with incremental progress saving
    num_well_addresses = len(tokenizer.well_address_tokenizer)

    from tqdm import tqdm

    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
        zarr_indices = batch["zarr_index"]

        if case == 2:
            # Compute embeddings for all well addresses
            embeddings = process_case2_batch(
                transformer_encoder, batch["meta"], num_well_addresses, device
            )
        else:
            # Standard embedding computation
            embeddings = compute_embeddings(transformer_encoder, batch["meta"], device)

        # Write to zarr
        batch_completed = []
        for i, idx in enumerate(zarr_indices):
            emb_zarr[idx] = embeddings[i]
            batch_completed.append(idx)

        # Update chunk parquet after each batch to save progress incrementally
        df = df.with_columns(
            pl.when(pl.col("zarr_index").is_in(batch_completed))
            .then(pl.lit(True))
            .otherwise(pl.col("complete"))
            .alias("complete")
        )
        df.write_parquet(chunk_path)

    # Verify all complete
    num_complete = df["complete"].sum()
    if df["complete"].all():
        log.info(f"All {num_complete} rows complete")
    else:
        remaining = (~df["complete"]).sum()
        log.warning(f"{remaining} rows still incomplete (completed {num_complete})")


def main():
    parser = argparse.ArgumentParser(description="Adapter embedding worker")
    parser.add_argument(
        "--worker_dir", type=str, required=True, help="Path to worker directory"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    run_worker(args.worker_dir, args.config)


if __name__ == "__main__":
    main()
