from typing import TYPE_CHECKING, Literal, Union

import ornamentalist
import torch
import torch.nn.functional as F
import torchdiffeq
from torch import nn

from hooke_forge.model.architecture import get_model_cls, get_tx_model_cls
from hooke_forge.model.context_encoders import (
    MetaDataConfig,
    ScalarEmbedder,
    TransformerEncoder,
    get_transformer_encoder,
)

if TYPE_CHECKING:
    from hooke_forge.model.drifting import JointDrifting
    from hooke_forge.model.mean_flow import JointMeanFlow


class JointFlowMatching(nn.Module):
    """Unified flow matching model supporting one or more modalities.

    Each modality has its own vector field, but the context encoder and
    time embedder are shared.  Works for both 4-D image tensors (Px) and
    2-D feature vectors (Tx) thanks to dimension-agnostic interpolation.

    Args:
        hidden_size: Dimensionality of the shared conditioning stream.
        context_encoder: TransformerEncoder that maps metadata -> conditioning vector.
        vector_fields: ``{modality_name: nn.Module}`` where each module has
            the signature ``(x, conditioning) -> prediction``.
    """

    def __init__(
        self,
        hidden_size: int,
        context_encoder: TransformerEncoder,
        vector_fields: dict[str, nn.Module],
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.t_embedder = ScalarEmbedder(hidden_size=hidden_size)
        self.vector_fields = nn.ModuleDict(vector_fields)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _forward_vector_field(
        self,
        modality: str,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        meta: dict[str, torch.Tensor],
        force_drop_rec_conc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)
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
        return self.vector_fields[modality](x, t_emb + meta_emb)

    def _classifier_free_guided_prediction(
        self,
        modality: str,
        x: torch.Tensor,
        t: torch.Tensor,
        meta: dict[str, torch.Tensor],
        cfg: float = 1.0,
    ) -> torch.Tensor:
        if t.ndim == 0:  # the ODE solver gives scalar t
            t = t.expand(x.shape[0])
        if cfg == 0.0:  # unconditional
            return self._forward_vector_field(
                modality,
                x=x,
                t=t,
                meta=meta,
                force_drop_rec_conc=torch.ones(x.shape[0], device=x.device, dtype=torch.long),
            )
        if cfg == 1.0:  # conditional
            return self._forward_vector_field(modality, x=x, t=t, meta=meta, force_drop_rec_conc=None)

        pred_cond = self._forward_vector_field(modality, x=x, t=t, meta=meta, force_drop_rec_conc=None)
        pred_null = self._forward_vector_field(
            modality,
            x=x,
            t=t,
            meta=meta,
            force_drop_rec_conc=torch.ones(x.shape[0], device=x.device, dtype=torch.long),
        )
        return pred_null + cfg * (pred_cond - pred_null)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def loss(
        self,
        modality: str,
        x0: torch.Tensor,  # sampled from source distribution
        x1: torch.Tensor,  # sampled from target distribution
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the flow matching loss for a single modality.

        Dimension-agnostic: works for (B, C, H, W) images or (B, D) vectors.
        """
        t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)
        t_ = t.reshape(-1, *([1] * (x0.ndim - 1))).expand_as(x0)

        xt = torch.lerp(x0, x1, t_)
        ut = x1 - x0

        vt = self._classifier_free_guided_prediction(modality, x=xt, t=t, meta=meta)
        return F.mse_loss(vt, ut)

    def forward(
        self,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """Compute losses for all provided modalities in a single DDP-safe forward.

        Args:
            batches: ``{modality: (x0, x1, meta)}``

        Returns:
            ``{modality: loss}``
        """
        return {m: self.loss(m, x0, x1, meta) for m, (x0, x1, meta) in batches.items()}

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        modality: str,
        x0: torch.Tensor,
        meta: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Generate a sample with the dopri5 probability flow ODE solver."""
        nfe = 0  # NB, if using guidance, the true nfe is this * 2

        def forward_fn(t, x):
            nonlocal nfe
            nfe += 1
            return self._classifier_free_guided_prediction(modality, x=x, t=t, meta=meta)

        traj = torchdiffeq.odeint(
            forward_fn,
            x0,
            torch.linspace(0, 1, 2, device=x0.device),
            method="dopri5",
            rtol=1e-5,
            atol=1e-5,
        )
        return traj[-1], nfe  # type: ignore


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@ornamentalist.configure(name="flow_model")
def get_model(
    approach: Literal["flow_matching", "drifting", "mean_flow"] = ornamentalist.Configurable["flow_matching"],
    modality: Literal["px", "tx", "joint"] = ornamentalist.Configurable["px"],
    tx_feature_dim: int = ornamentalist.Configurable[5000],  # Match HVG default
    metadata_config: MetaDataConfig = MetaDataConfig(),
    # Drifting-specific parameters
    tau: float = ornamentalist.Configurable[0.1],
    use_double_normalization: bool = ornamentalist.Configurable[True],
    use_feature_space: bool = ornamentalist.Configurable[False],
    multiple_temps: list[float] = (0.02, 0.05, 0.2),  # type: ignore  # drifting-only, not exposed to CLI
    cfg_weight: float = ornamentalist.Configurable[0.0],
    # MeanFlow-specific parameters (paper §3.1 & §3.2)
    mf_lognorm_mean: float = ornamentalist.Configurable[0.0],
    mf_lognorm_std: float = ornamentalist.Configurable[1.0],
    mf_r_neq_t_ratio: float = ornamentalist.Configurable[0.25],
    mf_adaptive_loss_power: float = ornamentalist.Configurable[1.0],
    mf_adaptive_loss_eps: float = ornamentalist.Configurable[1e-3],
    mf_cfg_omega: float = ornamentalist.Configurable[1.0],
    mf_cfg_kappa: float = ornamentalist.Configurable[0.0],
) -> Union[JointFlowMatching, "JointDrifting", "JointMeanFlow"]:
    """Build a generative model with the requested approach and modalities.

    The approach can be either "flow_matching" or "drifting".
    The DiT variant is controlled by ``--model.name`` (e.g. DiT-XL/2).
    The TX variant is controlled by ``--tx_model.name`` (e.g. TX-S).

    CLI examples::

        # Flow matching (existing)
        --flow_model.approach=flow_matching --flow_model.modality=px
        --flow_model.approach=flow_matching --flow_model.modality=joint

        # Drifting approach
        --flow_model.approach=drifting --flow_model.modality=px
        --flow_model.approach=drifting --flow_model.modality=joint --flow_model.tau=0.05

        # MeanFlow (new) — 1-NFE training from scratch
        --flow_model.approach=mean_flow --flow_model.modality=joint
        --flow_model.approach=mean_flow --flow_model.modality=joint \\
            --flow_model.mf_r_neq_t_ratio=0.25 \\
            --flow_model.mf_adaptive_loss_power=1.0 \\
            --flow_model.mf_cfg_omega=2.0
    """
    dit_cls = get_model_cls()  # uses --model.name config
    hidden_size: int = dit_cls.keywords["hidden_size"]  # type: ignore[union-attr]

    vector_fields: dict[str, nn.Module] = {}
    if modality in ("px", "joint"):
        vector_fields["px"] = dit_cls(
            input_size=32,
            in_channels=8,
            learn_sigma=False,
        )
    if modality in ("tx", "joint"):
        tx_cls = get_tx_model_cls()  # uses --tx_model.name config
        vector_fields["tx"] = tx_cls(
            data_dim=tx_feature_dim,
            cond_dim=hidden_size,
        )

    context_encoder = get_transformer_encoder(
        hidden_size=hidden_size,
        metadata_config=metadata_config,
    )

    # Dispatch based on approach
    if approach == "flow_matching":
        return JointFlowMatching(
            hidden_size=hidden_size,
            context_encoder=context_encoder,
            vector_fields=vector_fields,
        )
    elif approach == "drifting":
        # Import here to avoid circular imports
        from hooke_forge.model.drifting import JointDrifting

        return JointDrifting(
            hidden_size=hidden_size,
            context_encoder=context_encoder,
            vector_fields=vector_fields,
            tau=tau,
            use_double_normalization=use_double_normalization,
            use_feature_space=use_feature_space,
            multiple_temps=multiple_temps,
            cfg_weight=cfg_weight,
        )
    elif approach == "mean_flow":
        from hooke_forge.model.mean_flow import JointMeanFlow

        return JointMeanFlow(
            hidden_size=hidden_size,
            context_encoder=context_encoder,
            vector_fields=vector_fields,
            lognorm_mean=mf_lognorm_mean,
            lognorm_std=mf_lognorm_std,
            r_neq_t_ratio=mf_r_neq_t_ratio,
            adaptive_loss_power=mf_adaptive_loss_power,
            adaptive_loss_eps=mf_adaptive_loss_eps,
            cfg_omega=mf_cfg_omega,
            cfg_kappa=mf_cfg_kappa,
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")
