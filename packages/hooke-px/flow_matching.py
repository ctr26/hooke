import ornamentalist
import torch
from torch import nn
from model import DiT, get_model_cls
import torchdiffeq
from context_encoders import get_transformer_encoder, TransformerEncoder, ScalarEmbedder, MetaDataConfig

class FlowMatching(nn.Module):
    def __init__(self, hidden_size: int, context_encoder: TransformerEncoder, vector_field: nn.Module):
        super().__init__()
        self.vector_field = vector_field
        self.context_encoder = context_encoder
        self.t_embedder = ScalarEmbedder(hidden_size=hidden_size)

    def _forward_vector_field(self, *, x: torch.Tensor, t: torch.Tensor, meta: dict[str, torch.Tensor], force_drop_rec_conc: torch.Tensor | None = None) -> torch.Tensor:
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
        return self.vector_field(x, t_emb + meta_emb)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        x0: torch.Tensor,  # shape (B, C, H, W), sampled from source distribution
        x1: torch.Tensor,  # shape (B, C, H, W), sampled from target distribution
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the flow matching loss."""
        t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)  # (B,) - [0,1)
        t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

        xt = torch.lerp(x0, x1, t_)
        ut = x1 - x0

        vt = self._classifier_free_guided_prediction(x=xt, t=t, meta=meta)
        return torch.nn.functional.mse_loss(vt, ut)

    def _classifier_free_guided_prediction(
        self,
        x,
        t,
        meta: dict[str, torch.Tensor],
        cfg: float = 1.0,
    ) -> torch.Tensor:  # fmt: off
        if t.ndim == 0:  # the ODE solver gives scalar t
            t = t.expand(x.shape[0])
        if cfg == 0.0:  # unconditional
            return self._forward_vector_field(x=x, t=t, meta=meta, 
                force_drop_rec_conc=torch.ones(x.shape[0], device=x.device, dtype=torch.long),
            )
        if cfg == 1.0:  # conditional
            return self._forward_vector_field(x=x, t=t, meta=meta, force_drop_rec_conc=None)

        pred_cond = self._forward_vector_field(x=x, t=t, meta=meta, force_drop_rec_conc=None)
        pred_null = self._forward_vector_field(x=x, t=t, meta=meta, 
            force_drop_rec_conc=torch.ones(x.shape[0], device=x.device, dtype=torch.long),
        )
        return pred_null + cfg * (pred_cond - pred_null)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        x0: torch.Tensor,  # shape (B, C, H, W), sampled from N(0, I)
        meta: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Generate a sample with the dopri5 probability flow ODE solver."""
        nfe = 0  # NB, if using guidance, the true nfe is this * 2

        def forward_fn(t, x):
            nonlocal nfe
            nfe += 1
            return self._classifier_free_guided_prediction(x=x, t=t, meta=meta)

        traj = torchdiffeq.odeint(
            forward_fn,
            x0,
            torch.linspace(0, 1, 2, device=x0.device),
            method="dopri5",
            rtol=1e-5,
            atol=1e-5,
        )
        return traj[-1], nfe  # type: ignore


class JointFlowMatching(FlowMatching):
    def __init__(self, hidden_size: int, vector_field: nn.Module, context_encoder: TransformerEncoder):
        super().__init__(hidden_size=hidden_size, vector_field=vector_field, context_encoder=context_encoder)
        

@ornamentalist.configure(name="px_model")
def get_px_flow_matching_model(
    dit_model: str = ornamentalist.Configurable["DiT-XL/2"],
    metadata_config: MetaDataConfig = MetaDataConfig(),
) -> FlowMatching:
    dit_cls = get_model_cls(dit_model)
    return FlowMatching(
        hidden_size=dit_cls.hidden_size,
        vector_field=dit_cls,
        context_encoder=get_transformer_encoder(hidden_size=dit_cls.hidden_size, metadata_config=metadata_config),
    )

@ornamentalist.configure(name="tx_model")
def get_tx_flow_matching_model(
    mlp_in_dim: int,
    mlp_out_dim: int,
    hidden_size: int,
    mlp_dropout: float = 0.0,
    metadata_config: MetaDataConfig = MetaDataConfig(),
) -> FlowMatching:
    mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_out_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
        )
    return FlowMatching(
        hidden_size=hidden_size,
        vector_field=mlp,
        context_encoder=get_transformer_encoder(hidden_size=hidden_size, metadata_config=metadata_config),
    )
