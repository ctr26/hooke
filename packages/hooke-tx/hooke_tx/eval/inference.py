"""Inference: run model over a dataloader and return DataFrames aligned with ground truth."""

from __future__ import annotations

import pickle
from typing import Any

from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

import torch.distributed as dist

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from hooke_tx.data.constants import (
    ASSAY,
    CELL_TYPE,
    EMPTY,
    EXPERIMENT,
    GENE_ONLY,
    GENE_PERT,
    MOL_ONLY,
    MOL_PERT,
    MULTI_PERT,
    NEG_CONTROL,
    PERTURBATIONS,
    POS_CONTROL,
    WELL,
)

METADATA_COLUMNS = [
    CELL_TYPE,
    EXPERIMENT,
    WELL,
    ASSAY,
    PERTURBATIONS,
    EMPTY,
    NEG_CONTROL,
    POS_CONTROL,
    GENE_ONLY,
    MOL_ONLY,
    MULTI_PERT,
    GENE_PERT,
    MOL_PERT,
]


def generate_batch(
    model: Any,
    batch: dict[str, Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Run generation for one batch: move to device, call model.generate, return prediction tensor."""
    batch = dict(batch)
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    out = model.generate(batch)
    
    return out if isinstance(out, torch.Tensor) else out[0]


def _gather_inference_results(
    pred_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gather pred_rows and target_rows from all ranks so every rank has the full result."""
    if not dist.is_initialized():
        return pred_rows, target_rows

    world_size = dist.get_world_size()
    device = torch.device(device) if isinstance(device, str) else device

    num_preds = len(pred_rows)
    if num_preds > 0:
        preds = np.stack([r["predicted_expression"] for r in pred_rows])
        targets = np.stack([r["target_expression"] for r in target_rows])
        D = preds.shape[1]
    else:
        preds = np.zeros((0, 0), dtype=np.float32)
        targets = np.zeros((0, 0), dtype=np.float32)
        D = 0

    # All-gather (n, D) per rank
    local_shape = torch.tensor([num_preds, D], device=device, dtype=torch.long)
    shapes_per_rank = [torch.zeros(2, device=device, dtype=torch.long) for _ in range(world_size)]

    dist.all_gather(shapes_per_rank, local_shape)

    num_preds_per_rank = [int(shapes_per_rank[r][0].item()) for r in range(world_size)]
    max_n = max(num_preds_per_rank)
    D = max(int(shapes_per_rank[r][1].item()) for r in range(world_size))

    if max_n == 0 and D == 0:
        return [], []

    # Pad and all-gather pred, target
    preds_padded = torch.zeros(max_n, D, device=device, dtype=torch.float32)
    targets_padded = torch.zeros(max_n, D, device=device, dtype=torch.float32)
    
    if num_preds > 0:
        preds_padded[:num_preds] = torch.from_numpy(preds).to(device)
        targets_padded[:num_preds] = torch.from_numpy(targets).to(device)
    
    preds_per_rank = [torch.zeros(max_n, D, device=device, dtype=torch.float32) for _ in range(world_size)]
    targets_per_rank = [torch.zeros(max_n, D, device=device, dtype=torch.float32) for _ in range(world_size)]
    
    dist.all_gather(preds_per_rank, preds_padded)
    dist.all_gather(targets_per_rank, targets_padded)

    # Metadata: serialize list of dicts (metadata only), gather byte lengths, pad, all_gather
    meta_only = [{col: row.get(col) for col in METADATA_COLUMNS} for row in pred_rows]
    meta_bytes = pickle.dumps(meta_only)
    local_meta_len = len(meta_bytes)
    local_meta_len_t = torch.tensor([local_meta_len], device=device, dtype=torch.long)
    meta_len_per_rank = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    
    dist.all_gather(meta_len_per_rank, local_meta_len_t)
    
    max_len = max(int(meta_len_per_rank[r].item()) for r in range(world_size))
    meta_bytes_padded = meta_bytes.ljust(max_len, b"\x00")[:max_len]
    meta_padded = torch.tensor(list(meta_bytes_padded), device=device, dtype=torch.uint8)
    meta_per_rank = [torch.zeros(max_len, device=device, dtype=torch.uint8) for _ in range(world_size)]
    
    dist.all_gather(meta_per_rank, meta_padded)

    # Rebuild full pred_rows, target_rows on every rank
    pred_rows_full: list[dict[str, Any]] = []
    target_rows_full: list[dict[str, Any]] = []
    for rank in range(world_size):
        num_preds = num_preds_per_rank[rank]
        preds_rank = preds_per_rank[rank][:num_preds].cpu().numpy()
        targets_rank = targets_per_rank[rank][:num_preds].cpu().numpy()
        raw = bytes(meta_per_rank[rank][: meta_len_per_rank[rank].item()].cpu().numpy().tobytes())
        meta_row = pickle.loads(raw)
        
        for idx in range(num_preds):
            pred_rows_full.append({**meta_row[idx], "predicted_expression": preds_rank[idx]})
            target_rows_full.append({**meta_row[idx], "target_expression": targets_rank[idx]})

    return pred_rows_full, target_rows_full


def run_inference(
    model: pl.LightningModule,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run model over dataloader; return (pred_df, target_df) with identical format and shape.
    Each rank processes its own batches (dataset splitting via DistributedSampler). Results
    are gathered so all ranks receive the full DataFrames.
    """
    model.eval()

    pred_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    show_progress = True
    if dist.is_initialized() and dist.get_rank() != 0:
        show_progress = False

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference...", disable=not show_progress):
            preds = generate_batch(model, batch, device).cpu().numpy()
            
            # ODE solver returns (T, B, D); take final time step
            if preds.ndim == 3:
                preds = preds[-1]

            targets = batch["x1"].cpu().numpy()
            metadatas = batch.get("metadata")

            num_preds = preds.shape[0]
            if num_preds != targets.shape[0] or num_preds != len(metadatas):
                raise ValueError(
                    f"Batch size mismatch: pred {num_preds}, target {targets.shape[0]}, metadata {len(metadatas)}"
                )

            for idx in range(num_preds):
                meta = metadatas[idx]
                meta = {col: meta.get(col) for col in METADATA_COLUMNS}
                pred_rows.append({**meta, "predicted_expression": preds[idx]})
                target_rows.append({**meta, "target_expression": targets[idx]})

    if dist.is_initialized():
        pred_rows, target_rows = _gather_inference_results(pred_rows, target_rows, device)

    pred_df = pd.DataFrame(pred_rows)
    target_df = pd.DataFrame(target_rows)
    
    return pred_df, target_df
