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
    IMG_SIZE,
    MetaVocab,
)
from model import get_model_cls
from main import (
    REC_ID_DIM,
    CONCENTRATION_DIM,
    CELL_TYPE_DIM,
    ASSAY_TYPE_DIM,
    EXPERIMENT_DIM,
    WELL_ADDRESS_DIM,
)
from trainer import generate
from utils.ema import KarrasEMA
from utils.encoders import DINOv2Detector, Phenom2Detector, PH2BFDetector, StabilityCPEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Feature dimensions
PHENOM_DIM = 1664
DINO_DIM = 1024


def strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod prefix from state dict keys (added by torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    return new_state_dict


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
        rec_id_dim=REC_ID_DIM,
        concentration_dim=CONCENTRATION_DIM,
        cell_type_dim=CELL_TYPE_DIM,
        experiment_dim=EXPERIMENT_DIM,
        assay_type_dim=ASSAY_TYPE_DIM,
        well_address_dim=WELL_ADDRESS_DIM,
        pad_length=tokenizer.pad_length,
    )

    # Create model with correct vocab sizes (exclude pad_length which is not a model param)
    model_cls = get_model_cls()
    vocab_dict = dataclasses.asdict(vocab)
    del vocab_dict["pad_length"]
    net = model_cls(
        input_size=32,
        in_channels=8,
        learn_sigma=False,
        **vocab_dict,
    )
    net.to(device)

    # Load EMA weights (strip _orig_mod prefix if checkpoint was saved with torch.compile)
    ema = KarrasEMA(net)
    ema_state = strip_orig_mod_prefix(state["ema"])
    ema.load_state_dict(ema_state)

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

    # Extract real image features
    # real_phenom = phenom(px1_flat).cpu().numpy()
    # real_dino = dino(cp2rgb(px1_flat.to(torch.uint8))).cpu().numpy()
    # if S > 1:
    #     real_phenom = real_phenom.reshape(B, S, PHENOM_DIM)
    #     real_dino = real_dino.reshape(B, S, DINO_DIM)
    # else:
    #     real_phenom = real_phenom.reshape(B, PHENOM_DIM)
    #     real_dino = real_dino.reshape(B, DINO_DIM)

    if num_samples > 1:
        # Generate multiple samples per image (DART-style)
        meta_rep = {k: v.repeat_interleave(num_samples, dim=0) for k, v in meta.items()}
        px0 = torch.randn(B * num_samples, 8, 32, 32, device=device)
        preds, _ = generate(model=model, x0=px0, meta1=meta_rep)
        preds = vae.decode(preds)

        # Extract features and reshape to (B, num_samples, dim)
        pred_phenom = phenom(preds).cpu().numpy().reshape(B, num_samples, PHENOM_DIM)
        #pred_dino = dino(cp2rgb(preds)).cpu().numpy().reshape(B, num_samples, DINO_DIM)
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
        #"real_phenom": real_phenom,
        #"real_dino": real_dino,
        "pred_phenom": pred_phenom,
        #"pred_dino": pred_dino,
        "pred_images": pred_images,
    }


def run_worker(worker_dir: str, config_path: str):
    """Main worker function."""
    chunk_path = Path(worker_dir) / "chunk.parquet"

    with open(config_path) as f:
        config = json.load(f)

    # Initialize ornamentalist
    if "model" not in config:
        config["model"] = {"name": "DiT-XL/2"}

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
    assay_type = df["assay_type"].unique()
    assert len(assay_type) == 1, "All rows must have the same assay type"

    assay_type = assay_type[0]
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
    if assay_type == "brightfield_3channel":
        phenom = PH2BFDetector(device=device)
        log.info("Using PH2BF Embedding")
    else:
        phenom = Phenom2Detector(device=device)
        log.info("Using Phenom2 Embedding")
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
    #real_phenom_zarr = zarr.open(os.path.join(zarr_dir, "real_phenom.zarr"), mode="r+")
    #real_dino_zarr = zarr.open(os.path.join(zarr_dir, "real_dino.zarr"), mode="r+")
    pred_phenom_zarr = zarr.open(os.path.join(zarr_dir, "pred_phenom.zarr"), mode="r+")
    #pred_dino_zarr = zarr.open(os.path.join(zarr_dir, "pred_dino.zarr"), mode="r+")
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
            #real_phenom_zarr[idx] = results["real_phenom"][i]
            #real_dino_zarr[idx] = results["real_dino"][i]
            pred_phenom_zarr[idx] = results["pred_phenom"][i]
            #pred_dino_zarr[idx] = results["pred_dino"][i]
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
