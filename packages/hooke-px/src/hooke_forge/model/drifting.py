"""Drifting Models for Generation.

Based on the paper: "Drifting from Diffusion: A New Generation Paradigm".
Implements the drifting approach as an alternative to flow matching.

Key differences from flow matching:
- No time evolution or linear interpolation
- Uses drift field V(x) with attraction/repulsion dynamics
- Single-step generation (1-NFE)
- Stop-gradient training formulation
"""

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from hooke_forge.model.context_encoders import TransformerEncoder, ScalarEmbedder
from hooke_forge.model.architecture import get_model_cls, get_tx_model_cls


class JointDrifting(nn.Module):
    """Unified drifting model supporting one or more modalities.

    Uses attention-based drift computation instead of flow matching dynamics.
    Each modality has its own vector field, but context encoder and time embedder
    are shared for consistency with flow matching architecture.

    The core idea is to compute a drift field V(x) that:
    - Attracts samples toward positive examples (data)
    - Repels samples from negative examples (generated)
    - Reaches equilibrium (V=0) when generated distribution matches data

    Args:
        hidden_size: Dimensionality of the shared conditioning stream.
        context_encoder: TransformerEncoder that maps metadata -> conditioning vector.
        vector_fields: {modality_name: nn.Module} where each module has
            the signature (x, conditioning) -> prediction.
        tau: Temperature parameter for kernel computation.
        use_double_normalization: Whether to use paper's double normalization.
        use_feature_space: Whether to compute drift in feature space.
    """

    def __init__(
        self,
        hidden_size: int,
        context_encoder: TransformerEncoder,
        vector_fields: dict[str, nn.Module],
        tau: float = 0.1,
        use_double_normalization: bool = True,
        use_feature_space: bool = False,
        multiple_temps: list[float] = None,
        cfg_weight: float = 0.0,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.t_embedder = ScalarEmbedder(hidden_size=hidden_size)
        self.vector_fields = nn.ModuleDict(vector_fields)
        self.tau = tau
        self.use_double_normalization = use_double_normalization
        self.use_feature_space = use_feature_space
        self.multiple_temps = multiple_temps or [0.02, 0.05, 0.2]
        self.cfg_weight = cfg_weight  # fraction of batch to use as unconditional negatives

        # Setup feature encoders if using feature space
        if self.use_feature_space:
            self.setup_feature_encoders()

    # ------------------------------------------------------------------
    # Core drift computation
    # ------------------------------------------------------------------

    def compute_drift_velocity(
        self,
        x: torch.Tensor,  # Generated samples [N, D]
        y_pos: torch.Tensor,  # Positive samples (data) [N_pos, D]
        y_neg: torch.Tensor,  # Negative samples (generated) [N_neg, D]
        tau: float = None,
    ) -> torch.Tensor:
        """Compute drift velocity using normalized kernels (Paper Algorithm 2)."""

        if tau is None:
            tau = self.tau

        # Flatten higher-dimensional tensors for distance computation
        x_flat = self._flatten_for_distance(x)
        y_pos_flat = self._flatten_for_distance(y_pos)
        y_neg_flat = self._flatten_for_distance(y_neg)

        # Compute pairwise distances
        dist_pos = torch.cdist(x_flat, y_pos_flat)  # [N, N_pos]
        dist_neg = torch.cdist(x_flat, y_neg_flat)  # [N, N_neg]

        # Add large value to diagonal for y_neg=x case (avoid self-interaction)
        if y_neg is x:
            dist_neg = dist_neg + torch.eye(dist_neg.shape[0], device=x.device) * 1e6

        # Temperature-scaled logits
        logits_pos = -dist_pos / tau  # [N, N_pos]
        logits_neg = -dist_neg / tau  # [N, N_neg]

        # Combine logits and apply normalization
        logits = torch.cat([logits_pos, logits_neg], dim=1)  # [N, N_pos + N_neg]

        if self.use_double_normalization:
            # Paper's double normalization: softmax over both dimensions
            A_row = logits.softmax(dim=-1)  # Normalize over y-axis
            A_col = logits.softmax(dim=-2)  # Normalize over x-axis
            A = torch.sqrt(A_row * A_col)   # Geometric mean
        else:
            # Simple softmax normalization
            A = logits.softmax(dim=-1)

        # Split attention weights
        A_pos, A_neg = torch.split(A, [y_pos.shape[0], y_neg.shape[0]], dim=1)

        # Weighted mean computation (mean-shift style)
        # Need to handle the original shape
        if x.dim() > 2:  # For images or higher-dim tensors
            # Reshape for batch matrix multiplication
            x_reshaped = x.view(x.shape[0], -1)
            y_pos_reshaped = y_pos.view(y_pos.shape[0], -1)
            y_neg_reshaped = y_neg.view(y_neg.shape[0], -1)

            drift_pos = A_pos @ y_pos_reshaped  # [N, D_flat]
            drift_neg = A_neg @ y_neg_reshaped  # [N, D_flat]

            # Reshape back to original shape
            drift = (drift_pos - drift_neg).view_as(x)
        else:
            # For 2D tensors (features)
            drift_pos = A_pos @ y_pos  # [N, D]
            drift_neg = A_neg @ y_neg  # [N, D]
            drift = drift_pos - drift_neg

        return drift

    def _flatten_for_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten tensor for distance computation while preserving batch dimension."""
        return x.view(x.shape[0], -1)

    def compute_drift_with_features(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
        feature_encoder: nn.Module,
    ) -> torch.Tensor:
        """Compute drift in feature space with multi-temperature."""

        # Extract features
        with torch.no_grad():  # Feature encoders are frozen
            phi_x = feature_encoder(x)
            phi_y_pos = feature_encoder(y_pos)
            phi_y_neg = feature_encoder(y_neg)

        total_drift = 0

        # Multi-temperature computation
        for tau in self.multiple_temps:
            # Feature normalization (per paper)
            S = self._compute_feature_scale(phi_x, phi_y_pos, phi_y_neg)
            phi_x_norm = phi_x / S
            phi_y_pos_norm = phi_y_pos / S
            phi_y_neg_norm = phi_y_neg / S

            # Compute drift in normalized feature space
            drift_feats = self.compute_drift_velocity(
                phi_x_norm, phi_y_pos_norm, phi_y_neg_norm, tau
            )

            # Drift normalization
            lambda_norm = self._compute_drift_scale(drift_feats)
            drift_feats_norm = drift_feats / (lambda_norm + 1e-8)  # Add eps for stability

            total_drift += drift_feats_norm

        return total_drift

    def _compute_feature_scale(self, phi_x, phi_y_pos, phi_y_neg):
        """Feature normalization scale (Appendix eq.)."""
        # Average distance should be sqrt(C) where C is feature dim
        C = phi_x.shape[-1]
        all_feats = torch.cat([phi_x, phi_y_pos, phi_y_neg], dim=0)
        avg_dist = torch.cdist(all_feats, all_feats).mean()
        return avg_dist / math.sqrt(C)

    def _compute_drift_scale(self, drift):
        """Drift normalization scale (Appendix eq.)."""
        # Normalize so E[||V||²/C] ≈ 1
        C = drift.shape[-1] if drift.dim() == 2 else drift.numel() // drift.shape[0]
        return torch.sqrt((drift.view(drift.shape[0], -1).norm(dim=-1)**2 / C).mean())

    # ------------------------------------------------------------------
    # Feature encoders (optional)
    # ------------------------------------------------------------------

    def setup_feature_encoders(self):
        """Initialize feature encoders based on paper recommendations."""
        # For now, use identity encoders
        # In future phases, this would setup ResNet-style MAE encoders
        self.image_feature_encoder = nn.Identity()
        self.tx_feature_encoder = nn.Identity()

    def get_feature_encoder(self, modality: str) -> nn.Module:
        """Get feature encoder for drift computation in feature space."""
        if modality == "px":
            return self.image_feature_encoder
        elif modality == "tx":
            return self.tx_feature_encoder
        else:
            raise ValueError(f"Unknown modality: {modality}")

    # ------------------------------------------------------------------
    # Forward and loss computation
    # ------------------------------------------------------------------

    def _forward_without_drift(
        self,
        modality: str,
        epsilon: torch.Tensor,
        meta: dict[str, torch.Tensor],
        force_drop_rec_conc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples without drift computation (for initial x).

        Args:
            force_drop_rec_conc: If provided, forces conditioning dropout for
                classifier-free guidance. Pass all-ones tensor to drop all conditioning.
        """
        # Use time=0 embedding (current state, no time evolution)
        t_emb = self.t_embedder(torch.zeros(epsilon.shape[0], device=epsilon.device))
        meta_emb = self.context_encoder(
            rec_id=meta["rec_id"],
            concentration=meta["concentration"],
            comp_mask=meta["comp_mask"],
            cell_type=meta["cell_type"],
            experiment_label=meta["experiment_label"],
            assay_type=meta["assay_type"],
            well_address=meta["well_address"],
            force_drop_rec_conc=force_drop_rec_conc,
        )
        cond = t_emb + meta_emb

        return self.vector_fields[modality](epsilon, cond)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def loss(
        self,
        modality: str,
        x0: torch.Tensor,  # noise (unused in drifting - source distribution is always Gaussian)
        x1: torch.Tensor,  # target data (positive samples)
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute drifting loss following paper's fixed-point formulation.

        Optionally mixes unconditional negatives for classifier-free guidance
        when cfg_weight > 0 (paper's training-time CFG approach).
        """

        # Generate samples from noise (this is what gets optimized)
        epsilon = torch.randn_like(x1)
        x = self._forward_without_drift(modality, epsilon, meta)  # f_θ(ε)

        # Use x1 (target data) as positive examples
        # Use x (current generated samples) as negative examples
        y_pos = x1
        y_neg = x

        # Optional: mix in unconditional negatives for classifier-free guidance.
        # Per paper: weight unconditional samples by cfg_weight as additional negatives.
        if self.cfg_weight > 0.0:
            drop_all = torch.ones(epsilon.shape[0], device=epsilon.device, dtype=torch.long)
            x_uncond = self._forward_without_drift(modality, epsilon, meta, force_drop_rec_conc=drop_all)
            # Replicate unconditional samples proportionally to cfg_weight
            n_uncond = max(1, round(self.cfg_weight * epsilon.shape[0]))
            x_uncond_sample = x_uncond[:n_uncond]
            y_neg = torch.cat([x, x_uncond_sample], dim=0)

        # Compute drift field
        if self.use_feature_space:
            feature_encoder = self.get_feature_encoder(modality)
            drift_v = self.compute_drift_with_features(x, y_pos, y_neg, feature_encoder)
        else:
            drift_v = self.compute_drift_velocity(x, y_pos, y_neg)

        # CRITICAL: Stop-gradient formulation from paper
        x_drifted = (x + drift_v).detach()  # stopgrad(f_θ(ε) + V(f_θ(ε)))

        # Loss: minimize ||f_θ(ε) - stopgrad(f_θ(ε) + V(f_θ(ε)))||²
        # This is equivalent to minimizing ||V||² but with proper gradients
        loss = F.mse_loss(x, x_drifted)

        return loss

    def forward(
        self,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """Compute losses for all provided modalities in a single DDP-safe forward.

        Args:
            batches: {modality: (x0, x1, meta)}

        Returns:
            {modality: loss}
        """
        return {m: self.loss(m, x0, x1, meta) for m, (x0, x1, meta) in batches.items()}

    # ------------------------------------------------------------------
    # Generation (single-step, 1-NFE)
    # ------------------------------------------------------------------

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        modality: str,
        x0: torch.Tensor,  # noise input
        meta: dict[str, torch.Tensor],
        cfg: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Generate samples using single-step generation (1-NFE).

        Args:
            cfg: Guidance scale. cfg=1.0 means fully conditional (default).
                 cfg=0.0 means fully unconditional. Unlike flow matching,
                 drifting preserves 1-NFE even with cfg=0.0 (no extra forward pass).
        """
        force_drop = (
            torch.ones(x0.shape[0], device=x0.device, dtype=torch.long)
            if cfg == 0.0
            else None
        )
        x = self._forward_without_drift(modality, x0, meta, force_drop_rec_conc=force_drop)

        return x, 1  # Always 1 NFE (function evaluation)