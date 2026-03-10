"""Synthetic dataset factories for the MVP pipeline.

Produces random tensors matching hooke-forge's expected batch format:
- x0 (source), x1 (target): [B, D] tensors
- meta: dict of [B] integer tensors (tokenized metadata)

No real data is needed -- this is purely for testing pipeline wiring.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    """Generates random (x0, x1, meta) tuples."""

    def __init__(self, num_samples: int = 64, input_dim: int = 32, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.input_dim = input_dim
        gen = torch.Generator().manual_seed(seed)
        self.x0 = torch.randn(num_samples, input_dim, generator=gen)
        self.x1 = torch.randn(num_samples, input_dim, generator=gen)
        self.labels = torch.randint(0, 10, (num_samples,), generator=gen)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x0": self.x0[idx],
            "x1": self.x1[idx],
            "meta": {"label": self.labels[idx]},
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Custom collate that stacks the nested meta dict."""
    return {
        "x0": torch.stack([b["x0"] for b in batch]),
        "x1": torch.stack([b["x1"] for b in batch]),
        "meta": {"label": torch.stack([b["meta"]["label"] for b in batch])},
    }


def make_dataloaders(
    batch_size: int = 8,
    num_samples: int = 64,
    input_dim: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with synthetic data."""
    train_ds = SyntheticDataset(num_samples=num_samples, input_dim=input_dim, seed=seed)
    val_ds = SyntheticDataset(num_samples=max(num_samples // 4, 8), input_dim=input_dim, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
