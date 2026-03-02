"""Worker script for distributed inference.

Each worker processes a chunk of observations assigned to it, generates
predictions for the requested modality (px or tx), extracts representations,
and writes results to shared zarr arrays.

Usage:
    python run_inference_worker.py --worker_dir /path/to/worker_0 --config /path/to/config.json
"""

import argparse
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

from hooke_forge.model.tokenizer import DataFrameTokenizer
from hooke_forge.data.dataset import CellDataset, CellPaintConverter, IMG_SIZE, TxDataset
from hooke_forge.utils.ema import KarrasEMA
from hooke_forge.utils.encoders import (
    DINOv2Detector,
    Phenom2Detector,
    PH2BFDetector,
    StabilityCPEncoder,
    TxAMEncoder,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Representation registry
# ---------------------------------------------------------------------------

REPRESENTATION_DIMS: dict[str, tuple[tuple[int, ...], str]] = {
    "pred_phenom": ((1664,), "float32"),
    "pred_dino": ((1024,), "float32"),
    "pred_images": ((6, 256, 256), "uint8"),
    "pred_tx": (None, "float32"),  # dim set at runtime from model
    "pred_txam": ((512,), "float32"),
    "real_phenom": ((1664,), "float32"),
    "real_dino": ((1024,), "float32"),
}

DEFAULT_PX_REPRESENTATIONS = ["pred_phenom", "pred_images"]
DEFAULT_TX_REPRESENTATIONS = ["pred_tx"]

TX_ASSAY_TYPES = {"trek"}


def detect_modality(assay_type: str) -> str:
    return "tx" if assay_type.lower() in TX_ASSAY_TYPES else "px"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod prefix from state dict keys (added by torch.compile)."""
    return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}


def load_model(checkpoint_path: str, device: torch.device):
    """Load the model from checkpoint using get_model() factory.

    Returns (model, tokenizer).
    """
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)
    tokenizer = DataFrameTokenizer.from_state_dict(state["tokenizer"])

    from hooke_forge.model.flow_matching import get_model

    model = get_model()
    model.to(device)

    ema = KarrasEMA(model)
    ema.load_state_dict(strip_orig_mod_prefix(state["ema"]))
    model = ema.module
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Extractor builders
# ---------------------------------------------------------------------------


def build_extractors(representations: list[str], assay_type: str, device: torch.device) -> dict:
    """Build only the extractors needed for the requested representations.

    Returns {name: callable} dict.
    """
    extractors = {}

    # VAE is needed for any px representation that requires decoded images
    needs_vae = any(r in representations for r in ("pred_images", "pred_phenom", "pred_dino"))
    if needs_vae:
        extractors["_vae"] = StabilityCPEncoder(device=device)

    needs_phenom = any(r in representations for r in ("pred_phenom", "real_phenom"))
    if needs_phenom:
        if assay_type == "brightfield_3channel":
            extractors["_phenom"] = PH2BFDetector(device=device)
            log.info("Using PH2BF Embedding")
        else:
            extractors["_phenom"] = Phenom2Detector(device=device)
            log.info("Using Phenom2 Embedding")

    needs_dino = any(r in representations for r in ("pred_dino", "real_dino"))
    if needs_dino:
        extractors["_dino"] = DINOv2Detector(device=device)
        extractors["_cp2rgb"] = CellPaintConverter(device=device)

    if "pred_txam" in representations:
        extractors["_txam"] = TxAMEncoder(device=device)

    return extractors


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


@torch.inference_mode()
def process_batch_px(
    *,
    model,
    batch: dict,
    device: torch.device,
    extractors: dict,
    representations: list[str],
    num_samples: int,
) -> dict:
    """Process a px batch: generate latents → VAE decode → extract representations."""
    px1 = batch["img"].to(device, non_blocking=True)
    meta = {k: v.to(device, non_blocking=True) for k, v in batch["meta"].items()}
    zarr_indices = batch["zarr_index"].numpy()
    B = px1.shape[0]

    vae = extractors.get("_vae")
    phenom = extractors.get("_phenom")
    dino = extractors.get("_dino")
    cp2rgb = extractors.get("_cp2rgb")

    # Generate predictions
    if num_samples > 1:
        meta_rep = {k: v.repeat_interleave(num_samples, dim=0) for k, v in meta.items()}
        px0 = torch.randn(B * num_samples, 8, 32, 32, device=device)
        preds_latent, _ = model.generate("px", x0=px0, meta=meta_rep)
    else:
        px0 = torch.randn(B, 8, 32, 32, device=device)
        preds_latent, _ = model.generate("px", x0=px0, meta=meta)

    # Decode latents to images if needed
    needs_decode = any(r in representations for r in ("pred_images", "pred_phenom", "pred_dino"))
    preds = vae.decode(preds_latent) if needs_decode else None

    results = {"zarr_indices": zarr_indices}

    # Extract real image features
    if "real_phenom" in representations:
        real_feats = phenom(px1).cpu().numpy()
        results["real_phenom"] = real_feats

    if "real_dino" in representations:
        real_dino = dino(cp2rgb(px1.to(torch.uint8))).cpu().numpy()
        results["real_dino"] = real_dino

    # Extract predicted features
    if "pred_phenom" in representations:
        feat = phenom(preds).cpu().numpy()
        if num_samples > 1:
            feat = feat.reshape(B, num_samples, -1)
        results["pred_phenom"] = feat

    if "pred_dino" in representations:
        feat = dino(cp2rgb(preds)).cpu().numpy()
        if num_samples > 1:
            feat = feat.reshape(B, num_samples, -1)
        results["pred_dino"] = feat

    if "pred_images" in representations:
        imgs = preds.cpu().numpy()
        if num_samples > 1:
            imgs = imgs.reshape(B, num_samples, *imgs.shape[1:])
        results["pred_images"] = imgs

    return results


@torch.inference_mode()
def process_batch_tx(
    *,
    model,
    batch: dict,
    device: torch.device,
    extractors: dict,
    representations: list[str],
    num_samples: int,
    tx_feature_dim: int,
) -> dict:
    """Process a tx batch: generate tx vectors → extract representations."""
    meta = {k: v.to(device, non_blocking=True) for k, v in batch["meta"].items()}
    zarr_indices = batch["zarr_index"].numpy()
    B = meta["rec_id"].shape[0]

    if num_samples > 1:
        meta_rep = {k: v.repeat_interleave(num_samples, dim=0) for k, v in meta.items()}
        tx0 = torch.randn(B * num_samples, tx_feature_dim, device=device)
        preds, _ = model.generate("tx", x0=tx0, meta=meta_rep)
    else:
        tx0 = torch.randn(B, tx_feature_dim, device=device)
        preds, _ = model.generate("tx", x0=tx0, meta=meta)

    results = {"zarr_indices": zarr_indices}

    if "pred_tx" in representations:
        tx_out = preds.cpu().numpy()
        if num_samples > 1:
            tx_out = tx_out.reshape(B, num_samples, -1)
        results["pred_tx"] = tx_out

    if "pred_txam" in representations:
        txam = extractors["_txam"]
        feat = txam(preds).cpu().numpy()
        if num_samples > 1:
            feat = feat.reshape(B, num_samples, -1)
        results["pred_txam"] = feat

    return results


# ---------------------------------------------------------------------------
# Worker config & main loop
# ---------------------------------------------------------------------------


@ornamentalist.configure()
def get_worker_config(
    checkpoint_path: str = ornamentalist.Configurable[""],
    zarr_dir: str = ornamentalist.Configurable[""],
    num_samples_per_image: int = ornamentalist.Configurable[1],
    num_real_image_samples: int = ornamentalist.Configurable[1],
    batch_size: int = ornamentalist.Configurable[4],
    num_workers: int = ornamentalist.Configurable[4],
    representations: list = ornamentalist.Configurable[[]],
    tx_zarr_path: str = ornamentalist.Configurable[""],
):
    return (
        checkpoint_path,
        zarr_dir,
        num_samples_per_image,
        num_real_image_samples,
        batch_size,
        num_workers,
        representations,
        tx_zarr_path,
    )


def run_worker(worker_dir: str, config_path: str):
    """Main worker function."""
    chunk_path = Path(worker_dir) / "chunk.parquet"

    with open(config_path) as f:
        config = json.load(f)

    if "model" not in config:
        config["model"] = {"name": "DiT-XL/2"}

    ornamentalist.setup(config, force=True)

    (
        checkpoint_path,
        zarr_dir,
        num_samples,
        num_real_samples,
        batch_size,
        num_workers,
        representations,
        tx_zarr_path,
    ) = get_worker_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info(f"Worker directory: {worker_dir}")
    log.info(f"Num samples per image: {num_samples}")

    # Load chunk metadata
    df = pl.read_parquet(chunk_path)
    assay_col = "assay_type" if "assay_type" in df.columns else "image_type"
    assay_type = df[assay_col].unique()
    assert len(assay_type) == 1, "All rows must have the same assay type"
    assay_type = assay_type[0]

    modality = detect_modality(assay_type)
    log.info(f"Detected modality: {modality} (assay_type={assay_type})")

    # Default representations if not specified
    if not representations:
        representations = DEFAULT_TX_REPRESENTATIONS if modality == "tx" else DEFAULT_PX_REPRESENTATIONS
    log.info(f"Representations: {representations}")

    incomplete_mask = ~df["complete"]
    df_incomplete = df.filter(incomplete_mask)

    if len(df_incomplete) == 0:
        log.info("All rows already complete")
        return

    log.info(f"Processing {len(df_incomplete)} incomplete rows out of {len(df)} total")

    # Load model
    log.info(f"Loading checkpoint: {checkpoint_path}")
    model, tokenizer = load_model(checkpoint_path, device)

    # Build extractors
    extractors = build_extractors(representations, assay_type, device)

    # Create dataset and loader
    if modality == "tx":
        dataset_obj = TxDataset(
            metadata=df_incomplete,
            tokenizer=tokenizer,
            zarr_path=tx_zarr_path,
            train=False,
        )
    else:
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

    # Open shared zarr arrays for requested representations
    zarr_arrays = {}
    for name in representations:
        zarr_path = os.path.join(zarr_dir, f"{name}.zarr")
        zarr_arrays[name] = zarr.open(zarr_path, mode="r+")

    # Determine tx_feature_dim from model if needed
    tx_feature_dim = None
    if modality == "tx":
        tx_vf = model.vector_fields.get("tx")
        if tx_vf is not None:
            tx_feature_dim = tx_vf.data_dim
        else:
            cfg = ornamentalist.get_config()
            tx_feature_dim = cfg.get("flow_model", {}).get("tx_feature_dim", 5000)

    # Process batches
    completed_indices = []
    for batch in tqdm(loader, desc="Processing batches"):
        if modality == "tx":
            results = process_batch_tx(
                model=model,
                batch=batch,
                device=device,
                extractors=extractors,
                representations=representations,
                num_samples=num_samples,
                tx_feature_dim=tx_feature_dim,
            )
        else:
            results = process_batch_px(
                model=model,
                batch=batch,
                device=device,
                extractors=extractors,
                representations=representations,
                num_samples=num_samples,
            )

        # Write to zarr
        for i, idx in enumerate(results["zarr_indices"]):
            for name in representations:
                if name in results:
                    zarr_arrays[name][idx] = results[name][i]
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

    if df["complete"].all():
        log.info("All rows complete")
    else:
        remaining = (~df["complete"]).sum()
        log.warning(f"{remaining} rows still incomplete")


def main():
    parser = argparse.ArgumentParser(description="Distributed inference worker")
    parser.add_argument("--worker_dir", type=str, required=True, help="Path to worker directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    run_worker(args.worker_dir, args.config)


if __name__ == "__main__":
    main()
