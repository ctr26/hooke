"""MeanFlow: one-step generative model based on average velocity fields.

Implements the MeanFlow framework described in:
    "MeanFlow: Principled and Effective One-Step Generative Models"

Core idea
---------
Standard Flow Matching trains a network to predict the *instantaneous* velocity
``v(z_t, t)`` at each point along the flow path.  MeanFlow instead targets the
*average* velocity

    u(z_t, r, t)  =  (1 / (t − r)) · ∫_r^t  v(z_τ, τ) dτ

which encodes the displacement from ``r`` to ``t`` in a single quantity.  A
single evaluation of ``u_θ(ε, r=0, t=1)`` at inference generates a sample —
no ODE solver needed.

Training uses the **MeanFlow Identity**:

    u(z_t, r, t)  =  v(z_t, t)  −  (t − r) · (du/dt)

where the total time derivative is

    du/dt  =  v(z_t, t) · ∂_z u  +  ∂_t u

and is computed via a single JVP (``torch.func.jvp``) with tangents
``(v_t, 0, 1)`` along ``(z, r, t)``.  Stop-gradient is applied to the
target ``u_tgt``, so no higher-order derivatives are needed during the
θ-backward pass.

Classifier-free guidance (CFG) is baked into the training target (§3.2) so
that 1-NFE sampling is preserved even when guidance is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.func import functional_call, jvp

from hooke_forge.model.context_encoders import (
    TransformerEncoder,
    ScalarEmbedder,
    MetaDataConfig,
    get_transformer_encoder,
)
from hooke_forge.model.architecture import get_model_cls, get_tx_model_cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit_normal_sample(
    n: int,
    mean: float,
    std: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample n values from a logit-normal distribution onto (0, 1)."""
    return torch.sigmoid(torch.randn(n, device=device, dtype=dtype) * std + mean)


def _sample_t_r(
    batch_size: int,
    device: torch.device,
    lognorm_mean: float = 0.0,
    lognorm_std: float = 1.0,
    r_neq_t_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample time pairs (t, r) with t >= r.

    Strategy (paper §3.1, ablation Table 1a & 1d):
      1. Draw two independent logit-normal samples; assign the larger to t.
      2. With probability (1 − r_neq_t_ratio) set r = t, reducing to a
         standard FM step on those samples.
    """
    a = _logit_normal_sample(batch_size, lognorm_mean, lognorm_std, device)
    b = _logit_normal_sample(batch_size, lognorm_mean, lognorm_std, device)
    t = torch.maximum(a, b)
    r = torch.minimum(a, b)
    # Collapse r -> t for the (1 - ratio) fraction
    collapse = torch.rand(batch_size, device=device) >= r_neq_t_ratio
    r = torch.where(collapse, t, r)
    return t, r


def _adaptive_weight(
    error: torch.Tensor,
    power: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Per-sample adaptive loss weight  w = 1 / (‖Δ‖² + c)^p  (ECT / iCT).

    Gradient is detached so w is treated as a constant during θ-backprop.
    Shape: (B, 1, 1, ...) ready for broadcasting with ``error``.
    """
    sq_norm = error.detach().pow(2).flatten(1).sum(1)      # (B,)
    w = 1.0 / (sq_norm + eps).pow(power)                   # (B,)
    # Reshape for broadcasting against (B, ...)
    return w.reshape(-1, *([1] * (error.ndim - 1)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class JointMeanFlow(nn.Module):
    """Average-velocity generative model supporting one or more modalities.

    The network ``u_θ(z, r, t)`` is conditioned on *two* time variables.
    Following the paper's ablation (best result: embed ``(t, Δt)`` where
    ``Δt = t − r``), we maintain two ``ScalarEmbedder`` modules whose outputs
    are summed together with the metadata embedding before being fed into the
    existing DiT/TX ``vector_fields`` — no architecture changes required.

    Args:
        hidden_size: Shared conditioning stream width.
        context_encoder: Metadata → conditioning vector.
        vector_fields: ``{modality: nn.Module}`` with signature
            ``(z, cond) → prediction``.
        lognorm_mean: Mean of the logit-normal sampler (default 0).
        lognorm_std: Std of the logit-normal sampler (default 1).
        r_neq_t_ratio: Fraction of batch where r ≠ t (default 0.25).
        adaptive_loss_power: Exponent ``p`` in adaptive weight (default 1.0).
        adaptive_loss_eps: Stabiliser ``c`` in adaptive weight denominator.
        cfg_omega: CFG guidance scale ω baked into training target (1 = no CFG).
        cfg_kappa: Improved-CFG mixing scale κ (§A.1); 0 disables.
        uncond_drop_prob: Probability of dropping condition for CFG training.
    """

    def __init__(
        self,
        hidden_size: int,
        context_encoder: TransformerEncoder,
        vector_fields: dict[str, nn.Module],
        lognorm_mean: float = 0.0,
        lognorm_std: float = 1.0,
        r_neq_t_ratio: float = 0.25,
        adaptive_loss_power: float = 1.0,
        adaptive_loss_eps: float = 1e-3,
        cfg_omega: float = 1.0,
        cfg_kappa: float = 0.0,
        uncond_drop_prob: float = 0.10,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        # Two embedders: one for t, one for Δt = t − r (paper §3.1, Table 1c)
        self.t_embedder = ScalarEmbedder(hidden_size=hidden_size)
        self.dt_embedder = ScalarEmbedder(hidden_size=hidden_size)
        self.vector_fields = nn.ModuleDict(vector_fields)

        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.r_neq_t_ratio = r_neq_t_ratio
        self.adaptive_loss_power = adaptive_loss_power
        self.adaptive_loss_eps = adaptive_loss_eps
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.uncond_drop_prob = uncond_drop_prob

    # ------------------------------------------------------------------
    # Internal: conditioning and u_θ forward
    # ------------------------------------------------------------------

    def _cond(
        self,
        t: torch.Tensor,
        r: torch.Tensor,
        meta: dict[str, torch.Tensor],
        force_drop_rec_conc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build adaLN conditioning vector  c = emb(t) + emb(Δt) + emb(meta)."""
        return (
            self.t_embedder(t)
            + self.dt_embedder(t - r)
            + self.context_encoder(
                rec_id=meta["rec_id"],
                concentration=meta["concentration"],
                comp_mask=meta["comp_mask"],
                cell_type=meta["cell_type"],
                experiment_label=meta["experiment_label"],
                assay_type=meta["assay_type"],
                well_address=meta["well_address"],
                force_drop_rec_conc=force_drop_rec_conc,
            )
        )

    def _forward_u(
        self,
        modality: str,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        meta: dict[str, torch.Tensor],
        force_drop_rec_conc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate u_θ(z, r, t).  Returns tensor with same shape as z."""
        cond = self._cond(t, r, meta, force_drop_rec_conc)
        return self.vector_fields[modality](z, cond)

    # ------------------------------------------------------------------
    # JVP computation  —  Algorithm 1, line:  u, dudt = jvp(fn, ...)
    # ------------------------------------------------------------------

    def _jvp_step(
        self,
        modality: str,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v_eff: torch.Tensor,
        meta: dict[str, torch.Tensor],
        force_drop_rec_conc: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute  (u, du/dt)  via  torch.func.jvp.

        JVP tangent vector is  (v_eff, 0, 1)  along  (z, r, t), giving:
            du/dt  =  v_eff · ∂_z u  +  ∂_t u          (Eq. 7 in paper)

        The result ``dudt`` is subject to stop-gradient in the caller, so
        this backward pass does NOT feed into the θ-optimiser — no
        higher-order derivatives.
        """
        # Snapshot params/buffers once; captured by the closure below.
        params = {k: v for k, v in self.named_parameters()}
        buffers = {k: v for k, v in self.named_buffers()}

        # Prefix helpers ---------------------------------------------------
        def _p(prefix: str) -> tuple[dict, dict]:
            n = len(prefix) + 1
            p = {k[n:]: v for k, v in params.items()  if k.startswith(prefix + ".")}
            b = {k[n:]: v for k, v in buffers.items() if k.startswith(prefix + ".")}
            return p, b

        # Capture everything the pure function will need
        t_emb_mod   = self.t_embedder
        dt_emb_mod  = self.dt_embedder
        ctx_mod     = self.context_encoder
        vf_mod      = self.vector_fields[modality]

        t_emb_p,  t_emb_b  = _p("t_embedder")
        dt_emb_p, dt_emb_b = _p("dt_embedder")
        ctx_p,    ctx_b    = _p("context_encoder")
        vf_key = f"vector_fields.{modality}"
        vf_p,     vf_b     = _p(vf_key)

        meta_copy = meta
        fdrc = force_drop_rec_conc

        def u_fn(z_: torch.Tensor, r_: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
            """Pure (stateless) function of (z, r, t) for JVP."""
            t_emb  = functional_call(t_emb_mod,  (t_emb_p,  t_emb_b),  (t_,))
            dt_emb = functional_call(dt_emb_mod, (dt_emb_p, dt_emb_b), (t_ - r_,))
            meta_emb = functional_call(
                ctx_mod, (ctx_p, ctx_b), (),
                kwargs=dict(
                    rec_id=meta_copy["rec_id"],
                    concentration=meta_copy["concentration"],
                    comp_mask=meta_copy["comp_mask"],
                    cell_type=meta_copy["cell_type"],
                    experiment_label=meta_copy["experiment_label"],
                    assay_type=meta_copy["assay_type"],
                    well_address=meta_copy["well_address"],
                    force_drop_rec_conc=fdrc,
                ),
            )
            cond = t_emb + dt_emb + meta_emb
            return functional_call(vf_mod, (vf_p, vf_b), (z_, cond))

        # Tangents: dz/dt = v_eff,  dr/dt = 0,  dt/dt = 1
        tangents = (v_eff, torch.zeros_like(r), torch.ones_like(t))

        # Disable flash attention for JVP - it doesn't support forward-mode AD
        # Use math fallback instead which supports all AD modes
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False
        ):
            u, dudt = jvp(u_fn, (z, r, t), tangents)
        return u, dudt

    # ------------------------------------------------------------------
    # CFG-modified effective velocity  (§3.2, Eq. 16 / improved Eq. 22)
    # ------------------------------------------------------------------

    def _build_cfg_v_eff(
        self,
        modality: str,
        z: torch.Tensor,
        t: torch.Tensor,
        v_t: torch.Tensor,
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build ṽ_t for CFG-baked training.

        Basic (κ=0, Eq. 16):
            ṽ_t = ω · v_t  +  (1 − ω) · u_θ(z, t, t)   [unconditional]

        Improved (κ>0, Eq. 22):
            ṽ_t = ω · v_t  +  κ · u_θ(z, t, t | cls)
                            +  (1 − ω − κ) · u_θ(z, t, t)
        """
        B = z.shape[0]
        omega, kappa = self.cfg_omega, self.cfg_kappa

        # Unconditional arm  (r = t  →  Δt = 0)
        u_uncond = self._forward_u(
            modality, z, r=t, t=t, meta=meta,
            force_drop_rec_conc=torch.ones(B, device=z.device, dtype=torch.long),
        )
        v_eff = omega * v_t + (1.0 - omega - kappa) * u_uncond

        if kappa > 0.0:
            u_cond = self._forward_u(modality, z, r=t, t=t, meta=meta)
            v_eff = v_eff + kappa * u_cond

        return v_eff

    # ------------------------------------------------------------------
    # Training loss  (Algorithm 1)
    # ------------------------------------------------------------------

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def loss(
        self,
        modality: str,
        x0: torch.Tensor,   # source (data)
        x1: torch.Tensor,   # target (not used directly — noise drawn fresh)
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """MeanFlow training loss for one modality.

        Follows Algorithm 1 in the paper exactly::

            t, r  = sample_t_r()
            e     = randn_like(x)
            z     = (1 − t) * x + t * e
            v     = e − x
            u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
            u_tgt = v − (t − r) * dudt
            error = u − stopgrad(u_tgt)
            loss  = metric(error)
        """
        B = x0.shape[0]
        device = x0.device

        # Sample (t, r)
        t, r = _sample_t_r(
            B, device,
            lognorm_mean=self.lognorm_mean,
            lognorm_std=self.lognorm_std,
            r_neq_t_ratio=self.r_neq_t_ratio,
        )

        # Build interpolant and conditional velocity
        eps = torch.randn_like(x0)
        t_  = t.reshape(-1, *([1] * (x0.ndim - 1)))   # (B, 1, ...)
        z   = (1.0 - t_) * x0 + t_ * eps              # z_t
        v_t = eps - x0                                  # conditional velocity

        # Effective velocity for target (with CFG if omega != 1)
        if self.cfg_omega != 1.0:
            # Uncond-drop handled via force_drop_rec_conc inside helper;
            # class cond is dropped with uncond_drop_prob in context_encoder.
            v_eff = self._build_cfg_v_eff(modality, z, t, v_t, meta)
        else:
            v_eff = v_t

        # JVP → u and du/dt
        u, dudt = self._jvp_step(modality, z, r, t, v_eff, meta)

        # MeanFlow Identity: u_tgt = ṽ − (t − r) · du/dt
        dt = (t - r).reshape(-1, *([1] * (x0.ndim - 1)))
        u_tgt = (v_eff - dt * dudt).detach()           # stop-gradient on target

        # Adaptive-weighted L2 loss
        error = u - u_tgt
        w = _adaptive_weight(error, power=self.adaptive_loss_power, eps=self.adaptive_loss_eps)
        return (w * error.pow(2)).mean()

    def forward(
        self,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """Compute losses for all provided modalities in one forward pass.

        Args:
            batches: ``{modality: (x0, x1, meta)}``

        Returns:
            ``{modality: scalar_loss}``
        """
        return {m: self.loss(m, x0, x1, meta) for m, (x0, x1, meta) in batches.items()}

    # ------------------------------------------------------------------
    # Sampling  (Algorithm 2)
    # ------------------------------------------------------------------

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        modality: str,
        x0: torch.Tensor,              # noise ε ~ p_prior
        meta: dict[str, torch.Tensor],
        cfg: float = 1.0,              # inference-time CFG override (doubles NFE)
        n_steps: int = 1,
    ) -> tuple[torch.Tensor, int]:
        """Generate a sample using the average-velocity field.

        1-step (default, Algorithm 2):
            x_0 = ε  −  u_θ(ε, r=0, t=1)

        N-step (few-step, optional):
            Iterates  z_r = z_t − (t − r) · u_θ(z_t, r, t)
            over a uniform grid on [0, 1].

        When ``cfg=1.0`` (default) guidance is already baked in from
        training — no extra NFE.  Setting ``cfg != 1.0`` applies
        inference-time linear combination (doubles NFE).

        Returns:
            (sample, nfe)
        """
        nfe = 0
        B = x0.shape[0]
        device = x0.device

        def _u(z: torch.Tensor, r_val: float, t_val: float) -> torch.Tensor:
            nonlocal nfe
            r = torch.full((B,), r_val, device=device, dtype=torch.float32)
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            if cfg == 1.0:
                nfe += 1
                return self._forward_u(modality, z, r=r, t=t, meta=meta)
            else:
                # Inference-time CFG  (doubles NFE)
                u_cond = self._forward_u(modality, z, r=r, t=t, meta=meta)
                u_null = self._forward_u(
                    modality, z, r=r, t=t, meta=meta,
                    force_drop_rec_conc=torch.ones(B, device=device, dtype=torch.long),
                )
                nfe += 2
                return u_null + cfg * (u_cond - u_null)

        if n_steps == 1:
            # Algorithm 2: single network evaluation
            u = _u(x0, r_val=0.0, t_val=1.0)
            sample = x0 - u                        # z_0 = z_1 − (1−0)·u
        else:
            # Few-step on uniform grid t=1 → t=0
            ts = torch.linspace(1.0, 0.0, n_steps + 1).tolist()
            z = x0
            for i in range(n_steps):
                t_val, r_val = ts[i], ts[i + 1]
                u = _u(z, r_val=r_val, t_val=t_val)
                z = z - (t_val - r_val) * u
            sample = z

        return sample, nfe
