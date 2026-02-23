from typing import Literal

import ornamentalist
import torch
from torch import nn
import torch.nn.functional as F
import torchdiffeq
from hooke_forge.model.context_encoders import get_transformer_encoder, TransformerEncoder, ScalarEmbedder, MetaDataConfig
from hooke_forge.model.architecture import get_model_cls, get_tx_model_cls


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
                modality, x=x, t=t, meta=meta,
                force_drop_rec_conc=torch.ones(x.shape[0], device=x.device, dtype=torch.long),
            )
        if cfg == 1.0:  # conditional
            return self._forward_vector_field(modality, x=x, t=t, meta=meta, force_drop_rec_conc=None)

        pred_cond = self._forward_vector_field(modality, x=x, t=t, meta=meta, force_drop_rec_conc=None)
        pred_null = self._forward_vector_field(
            modality, x=x, t=t, meta=meta,
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
    modality: Literal["px", "tx", "joint"] = ornamentalist.Configurable["px"],
    tx_feature_dim: int = ornamentalist.Configurable[1024],
    metadata_config: MetaDataConfig = MetaDataConfig(),
) -> JointFlowMatching:
    """Build a JointFlowMatching model with the requested modalities.

    The DiT variant is controlled by ``--model.name`` (e.g. DiT-XL/2).
    The TX variant is controlled by ``--tx_model.name`` (e.g. TX-S).

    CLI examples::

        --flow_model.modality=px                     # Px only (default)
        --flow_model.modality=tx --flow_model.tx_feature_dim=1024
        --flow_model.modality=joint
    """
    dit_cls = get_model_cls()  # uses --model.name config
    hidden_size: int = dit_cls.keywords["hidden_size"]  # type: ignore[union-attr]

    vector_fields: dict[str, nn.Module] = {}
    if modality in ("px", "joint"):
        vector_fields["px"] = dit_cls(
            input_size=32, in_channels=8, learn_sigma=False,
        )
    if modality in ("tx", "joint"):
        tx_cls = get_tx_model_cls()  # uses --tx_model.name config
        vector_fields["tx"] = tx_cls(
            data_dim=tx_feature_dim,
            cond_dim=hidden_size,
        )

    context_encoder = get_transformer_encoder(
        hidden_size=hidden_size, metadata_config=metadata_config,
    )
    return JointFlowMatching(
        hidden_size=hidden_size,
        context_encoder=context_encoder,
        vector_fields=vector_fields,
    )
