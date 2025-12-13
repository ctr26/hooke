"""Worker script for distributed inference.

Each worker processes a chunk of images assigned to it, computes embeddings,
and writes results to a shared zarr file.

Usage:
    python run_inference_worker.py --worker_dir /path/to/worker_0 --config /path/to/config.json
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import ornamentalist
import polars as pl
import torch
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm

from adaptor import DataFrameTokenizer
from dataset import (
    CellDataset,
    CellPaintConverter,
    StabilityCPEncoder,
    IMG_SIZE,
    MetaVocab,
)
from model import get_model_cls
from trainer import generate
from utils.ema import KarrasEMA
from utils.evaluation import DINOv2Detector, Phenom2Detector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Feature dimensions
PHENOM_DIM = 1664
DINO_DIM = 1024


@ornamentalist.configure()
def get_worker_config(
    checkpoint_path: str = ornamentalist.Configurable[""],
    zarr_dir: str = ornamentalist.Configurable[""],
    num_samples_per_image: int = ornamentalist.Configurable[1],
    num_real_image_samples: int = ornamentalist.Configurable[1],
    batch_size: int = ornamentalist.Configurable[4],
    num_workers: int = ornamentalist.Configurable[4],
):
    return (
        checkpoint_path,
        zarr_dir,
        num_samples_per_image,
        num_real_image_samples,
        batch_size,
        num_workers,
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
):
    """Load the model and VAE from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, vae, tokenizer)
    """
    # Load checkpoint
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # Restore tokenizer from checkpoint
    tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    # Build vocab from tokenizer
    vocab = MetaVocab(
        rec_id_dim=len(tokenizer.rec_id_tokenizer),
        concentration_dim=len(tokenizer.concentration_tokenizer),
        cell_type_dim=len(tokenizer.cell_type_tokenizer),
        experiment_dim=len(tokenizer.experiment_tokenizer),
        image_type_dim=len(tokenizer.image_type_tokenizer),
        well_address_dim=len(tokenizer.well_address_tokenizer),
        pad_length=tokenizer.pad_length,
    )

    # Create model with correct vocab sizes
    model_cls = get_model_cls()
    net = model_cls(
        input_size=32,
        in_channels=8,
        learn_sigma=False,
        **dataclasses.asdict(vocab),
    )
    net.to(device)

    # Load EMA weights
    ema = KarrasEMA(net)
    ema.load_state_dict(state["ema"])

    model = ema.module
    model.eval()

    vae = StabilityCPEncoder(device=device)
    return model, vae, tokenizer


@torch.inference_mode()
def process_batch(
    *,
    model,
    vae: StabilityCPEncoder,
    batch: dict,
    device: torch.device,
    phenom: Phenom2Detector,
    dino: DINOv2Detector,
    cp2rgb: CellPaintConverter,
    num_samples: int,
) -> dict:
    """Process a single batch and return features."""
    px1 = batch["img"].to(device, non_blocking=True)
    meta = {k: v.to(device, non_blocking=True) for k, v in batch["meta"].items()}
    zarr_indices = batch["zarr_index"].numpy()

    # Allow multiple real image crops per example (B, S, C, H, W)
    if px1.ndim == 5:
        B, S, C, H, W = px1.shape
        px1_flat = px1.reshape(B * S, C, H, W)
    else:
        B, C, H, W = px1.shape
        S = 1
        px1_flat = px1

    # Extract real image features (keep sample dimension if present)
    real_phenom = phenom(px1_flat).cpu().numpy().reshape(B, S, PHENOM_DIM)
    real_dino = (
        dino(cp2rgb(px1_flat.to(torch.uint8))).cpu().numpy().reshape(B, S, DINO_DIM)
    )

    if num_samples > 1:
        # Generate multiple samples per image (DART-style)
        meta_rep = {
            k: v.repeat_interleave(num_samples, dim=0) for k, v in meta.items()
        }
        px0 = torch.randn(B * num_samples, 8, 32, 32, device=device)
        preds, _ = generate(model=model, x0=px0, meta1=meta_rep)
        preds = vae.decode(preds)

        # Extract features and reshape to (B, num_samples, dim)
        pred_phenom = phenom(preds).cpu().numpy().reshape(B, num_samples, PHENOM_DIM)
        pred_dino = dino(cp2rgb(preds)).cpu().numpy().reshape(B, num_samples, DINO_DIM)
        pred_images = (
            preds.cpu()
            .numpy()
            .reshape(B, num_samples, preds.shape[1], preds.shape[2], preds.shape[3])
        )
    else:
        # Single sample per image
        px0 = torch.randn(B, 8, 32, 32, device=device)
        preds, _ = generate(model=model, x0=px0, meta1=meta)
        preds = vae.decode(preds)

        pred_phenom = phenom(preds).cpu().numpy()
        pred_dino = dino(cp2rgb(preds)).cpu().numpy()
        pred_images = preds.cpu().numpy()

    return {
        "zarr_indices": zarr_indices,
        "real_phenom": real_phenom,
        "real_dino": real_dino,
        "pred_phenom": pred_phenom,
        "pred_dino": pred_dino,
        "pred_images": pred_images,
    }


def run_worker(worker_dir: str, config_path: str):
    """Main worker function."""
    worker_dir = Path(worker_dir)
    chunk_path = worker_dir / "chunk.parquet"

    with open(config_path) as f:
        config = json.load(f)

    # Initialize ornamentalist
    if "get_model_cls" not in config:
        config["get_model_cls"] = {"name": "DiT-XL/2"}

    ornamentalist.setup(config, force=True)

    # Get injected parameters
    (
        checkpoint_path,
        zarr_dir,
        num_samples,
        num_real_samples,
        batch_size,
        num_workers,
    ) = get_worker_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Worker directory: {worker_dir}")
    log.info(f"Num samples per image: {num_samples}")

    # Load chunk metadata
    df = pl.read_parquet(chunk_path)
    incomplete_mask = ~df["complete"]
    df_incomplete = df.filter(incomplete_mask)

    if len(df_incomplete) == 0:
        log.info("All rows already complete")
        return

    log.info(f"Processing {len(df_incomplete)} incomplete rows out of {len(df)} total")

    # Load model (also returns tokenizer from checkpoint)
    log.info(f"Loading checkpoint: {checkpoint_path}")
    model, vae, tokenizer = load_model(checkpoint_path, device)

    # Initialize feature extractors
    phenom = Phenom2Detector(device=device)
    dino = DINOv2Detector(device=device)
    cp2rgb = CellPaintConverter(device=device)

    # Create dataset and loader for incomplete rows only
    dataset_obj = CellDataset(
        metadata=df_incomplete,
        tokenizer=tokenizer,
        train=False,
        size=IMG_SIZE,
    )
    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Open shared zarr arrays
    real_phenom_zarr = zarr.open(os.path.join(zarr_dir, "real_phenom.zarr"), mode="r+")
    real_dino_zarr = zarr.open(os.path.join(zarr_dir, "real_dino.zarr"), mode="r+")
    pred_phenom_zarr = zarr.open(os.path.join(zarr_dir, "pred_phenom.zarr"), mode="r+")
    pred_dino_zarr = zarr.open(os.path.join(zarr_dir, "pred_dino.zarr"), mode="r+")
    pred_images_zarr = zarr.open(os.path.join(zarr_dir, "pred_images.zarr"), mode="r+")

    # Process batches
    completed_indices = []
    for batch in tqdm(loader, desc="Processing batches"):
        results = process_batch(
            model=model,
            vae=vae,
            batch=batch,
            device=device,
            phenom=phenom,
            dino=dino,
            cp2rgb=cp2rgb,
            num_samples=num_samples,
        )

        # Write to zarr
        for i, idx in enumerate(results["zarr_indices"]):
            real_phenom_zarr[idx] = results["real_phenom"][i]
            real_dino_zarr[idx] = results["real_dino"][i]
            pred_phenom_zarr[idx] = results["pred_phenom"][i]
            pred_dino_zarr[idx] = results["pred_dino"][i]
            pred_images_zarr[idx] = results["pred_images"][i]
            completed_indices.append(idx)

    # Mark completed rows in the parquet file
    log.info(f"Marking {len(completed_indices)} rows as complete")
    df = df.with_columns(
        pl.when(pl.col("zarr_index").is_in(completed_indices))
        .then(pl.lit(True))
        .otherwise(pl.col("complete"))
        .alias("complete")
    )
    df.write_parquet(chunk_path)

    # Verify all complete
    if df["complete"].all():
        log.info("All rows complete")
    else:
        remaining = (~df["complete"]).sum()
        log.warning(f"{remaining} rows still incomplete")


def main():
    parser = argparse.ArgumentParser(description="Distributed inference worker")
    parser.add_argument(
        "--worker_dir", type=str, required=True, help="Path to worker directory"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    run_worker(args.worker_dir, args.config)


if __name__ == "__main__":
    main()
