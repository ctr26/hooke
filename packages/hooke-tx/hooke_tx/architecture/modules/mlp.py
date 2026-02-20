import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """MLP block with optional skip connection (none, sum, or cat)."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: bool = False,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
        skip_type: str = "none",  # none, "sum", or "cat"
    ):
        super().__init__()
        self.skip_type = skip_type
        self.use_norm = norm
        
        layers = []
        if self.use_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, out_dim))
        if activation is not None:
            layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
        
        # Setup skip connection components
        if self.skip_type == "cat":
            # Cat works regardless of dimensions - concatenate and project
            # Skip normalization is optional and follows the main norm setting
            if self.use_norm:
                self.skip_norm = nn.LayerNorm(out_dim + in_dim)
            else:
                self.skip_norm = None
            self.skip_linear = nn.Linear(out_dim + in_dim, out_dim)
        elif self.skip_type == "sum":
            # Sum only works when dimensions match (no projection)
            self.use_sum_skip = (in_dim == out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        
        if self.skip_type == "cat":
            # Cat works regardless of dimensions - just concatenate
            cat = torch.cat([out, x], dim=-1)
            
            # Apply normalization if enabled (follows main norm setting)
            if self.skip_norm is not None:
                cat = self.skip_norm(cat)
            
            return self.skip_linear(cat)

        elif self.skip_type == "sum":
            # Sum only works when dimensions match (no projection)
            if self.use_sum_skip:
                return out + x
        
        return out