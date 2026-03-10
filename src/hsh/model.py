"""Minimal flow-matching model for the MVP pipeline.

Mirrors hooke-forge's JointFlowMatching pattern:
- Takes source x0, target x1, and time t
- Learns to predict the velocity field v = x1 - x0
- Loss is MSE between predicted and true velocity

Deliberately tiny (2-layer MLP) so it trains in <1s on CPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hsh.config import ModelConfig


class FlowMatchingMLP(nn.Module):
    """Minimal flow-matching velocity predictor.

    Architecture: x_t concatenated with t -> MLP -> predicted velocity.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim + 1, config.hidden_dim),
            nn.SiLU(),
            *[
                layer
                for _ in range(config.num_layers - 1)
                for layer in (nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU())
            ],
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field at (x_t, t).

        Args:
            x_t: Interpolated state, shape [B, D].
            t: Time in [0, 1], shape [B] or [B, 1].
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t], dim=-1))

    def loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute flow-matching loss (MSE between predicted and true velocity)."""
        t = torch.rand(x0.shape[0], device=x0.device)
        t_expanded = t.unsqueeze(-1).expand_as(x0)

        x_t = torch.lerp(x0, x1, t_expanded)
        true_velocity = x1 - x0

        pred_velocity = self.forward(x_t, t)
        return F.mse_loss(pred_velocity, true_velocity)

    @torch.no_grad()
    def generate(self, x0: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Generate samples via Euler integration of the learned velocity field."""
        dt = 1.0 / num_steps
        x = x0.clone()
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.forward(x, t)
            x = x + v * dt
        return x
