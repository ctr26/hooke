import json

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hooke_tx.data.constants import EMPTY
from hooke_tx.data.chem_utils import (
    compute_ecfp,
    compute_inchikey,
    standardize_smiles,
)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Projection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Fourier(nn.Module):
    """
    Fourier embedding layer for time encoding.
    """
    def __init__(
        self,
        dim: int,
        bandwidth: int = 1,
    ):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(dim) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        target_device = y.device
        freqs = self.get_buffer("freqs").to(device=target_device, dtype=torch.float32)
        phases = self.get_buffer("phases").to(device=target_device, dtype=torch.float32)
        
        y = torch.ger(y, freqs)
        y = y + phases
        y = y.cos() * np.sqrt(2)
        
        return y.to(x.dtype)


class OneHotEmbedder(nn.Module):
    def __init__(
        self,
        all_labels: list[str],
        dim: int = 128,
    ):
        super().__init__()
        all_labels = sorted(all_labels)
        all_labels.append(EMPTY)

        self.embedding = nn.Embedding(len(all_labels), dim)
        self.label_to_id = {label: idx for idx, label in enumerate(all_labels)}

    def forward(self, batch_labels: list[str | float]) -> torch.Tensor:
        label_indices = [self.label_to_id[label] for label in batch_labels]
        device = self.embedding.weight.device
        
        return self.embedding(torch.tensor(label_indices, device=device, dtype=torch.long))


class ECFPEmbedder(nn.Module):
    """Embeds SMILES via ECFP fingerprints. Invalid SMILES use learnable fallback."""
    def __init__(
        self,
        all_labels: list[str],
        dim: int = 1024,
        radius: int = 2,
    ):
        super().__init__()
        labels = sorted(set(all_labels)) + [EMPTY]
        valid_fps, invalid_labels = [], []
        label_to_fp_idx = {}

        for label in labels:
            if label == EMPTY:
                invalid_labels.append(label)
                continue
            
            std_smiles = standardize_smiles(label)
            
            if not std_smiles:
                invalid_labels.append(label)
                logger.warning(f"ECFPEmbedder: invalid SMILES '{label}' -> learnable fallback")
                continue
            
            fp = compute_ecfp(std_smiles, embedding_dim=dim, radius=radius)
            
            if fp is None:
                invalid_labels.append(label)
                logger.warning(f"ECFPEmbedder: no fingerprint for '{label}' -> learnable fallback")
                continue
            
            label_to_fp_idx[label] = len(valid_fps)
            valid_fps.append(fp)

        self.label_to_fp_idx = label_to_fp_idx
        
        self.register_buffer(
            "fingerprint_matrix",
            torch.tensor(np.array(valid_fps) if valid_fps else np.zeros((0, dim), dtype=np.float32), dtype=torch.float32),
        )
        
        self.fallback_embedding = nn.Embedding(len(invalid_labels), dim)
        self.invalid_label_to_fallback_idx = {label: i for i, label in enumerate(invalid_labels)}

    def forward(self, batch_labels: list[str | float]) -> torch.Tensor:
        device = self.fingerprint_matrix.device
        
        out = []
        for label in batch_labels:
            label = str(label)
            
            if label in self.label_to_fp_idx:
                idx = self.label_to_fp_idx[label]
                out.append(self.fingerprint_matrix[idx : idx + 1])
            else:
                idx = self.invalid_label_to_fallback_idx.get(label, 0)
                out.append(self.fallback_embedding(torch.tensor([idx], device=device, dtype=torch.long)))
        
        return torch.cat(out, dim=0)


class MolGPSEmbedder(nn.Module):
    """Embeds SMILES via MolGPS cache lookup. Invalid/missing use learnable fallback."""
    def __init__(
        self,
        all_labels: list[str],
        emb_name: str,
        cache_dir: str = "/rxrx/data/valence/pef/molgps",
    ):
        super().__init__()
        labels = sorted(set(all_labels)) + [EMPTY]
        valid_fps, invalid_labels = [], []
        label_to_fp_idx = {}

        cached_embeddings = torch.load(f"{cache_dir}/embeddings.pt", map_location="cpu", weights_only=True)
        index_map = pd.read_parquet(f"{cache_dir}/index_map.parquet")
        
        smiles_to_idx = {s: i for i, s in enumerate(index_map["smiles"].values.tolist())}
        inchikey_to_idx = {ik: i for i, ik in enumerate(index_map["inchikey"].values.tolist())}
        
        shapes = json.load(open(f"{cache_dir}/shapes.json"))
        fp_dim = shapes[emb_name]

        for label in labels:
            if label == EMPTY:
                invalid_labels.append(label)
                continue
            
            std_smiles = standardize_smiles(label)
            
            if not std_smiles:
                invalid_labels.append(label)
                logger.warning(f"MolGPSEmbedder: invalid SMILES '{label}' -> learnable fallback")
                continue
            
            idx = smiles_to_idx.get(std_smiles)
            
            if idx is None:
                try:
                    inchikey = compute_inchikey(std_smiles, standardize=False)
                except Exception:
                    invalid_labels.append(label)
                    logger.warning(f"MolGPSEmbedder: failed InChIKey for '{label}' -> learnable fallback")
                    continue
                
                idx = inchikey_to_idx.get(inchikey)
            
            if idx is None:
                invalid_labels.append(label)
                logger.warning(f"MolGPSEmbedder: '{label}' not in cache -> learnable fallback")
                continue
            
            fp = np.array(cached_embeddings[emb_name][idx], dtype=np.float32)
            label_to_fp_idx[label] = len(valid_fps)
            valid_fps.append(fp)

        self.label_to_fp_idx = label_to_fp_idx
        
        self.register_buffer(
            "fingerprint_matrix",
            torch.tensor(np.array(valid_fps) if valid_fps else np.zeros((0, fp_dim), dtype=np.float32), dtype=torch.float32),
        )
        
        self.fallback_embedding = nn.Embedding(len(invalid_labels), fp_dim)
        self.invalid_label_to_fallback_idx = {label: i for i, label in enumerate(invalid_labels)}

    def forward(self, batch_labels: list[str | float]) -> torch.Tensor:
        device = self.fingerprint_matrix.device
        
        out = []
        for label in batch_labels:
            label = str(label)
            
            if label in self.label_to_fp_idx:
                idx = self.label_to_fp_idx[label]
                out.append(self.fingerprint_matrix[idx : idx + 1])
            else:
                idx = self.invalid_label_to_fallback_idx.get(label, 0)
                out.append(self.fallback_embedding(torch.tensor([idx], device=device, dtype=torch.long)))
        
        return torch.cat(out, dim=0)

