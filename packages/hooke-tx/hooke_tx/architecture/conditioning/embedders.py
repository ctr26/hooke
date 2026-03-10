import torch
import torch.nn as nn
import numpy as np

from hooke_tx.data.constants import EMPTY


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Projection(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Fourier(nn.Module):
    """
    Fourier embedding layer for time encoding.
    """
    def __init__(self, dim, bandwidth=1):
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
    def __init__(self, labels: list[str], dim: int):
        super().__init__()
        labels = sorted(labels)
        labels.append(EMPTY)

        self.embedding = nn.Embedding(len(labels), dim)
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}

    def forward(self, labels: list[str | float]) -> torch.Tensor:
        label_indices = [self.label_to_id[label] for label in labels]
        device = self.embedding.weight.device
        return self.embedding(torch.tensor(label_indices, device=device, dtype=torch.long))