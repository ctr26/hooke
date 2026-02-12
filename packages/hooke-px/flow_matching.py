import torch
from torch import nn
from model import DiTWrapper
import torchdiffeq
from trainer import DDP

class FlowMatching(nn.Module):
    def __init__(self, model: DiTWrapper):
        super().__init__()
        self.model = model

    def forward(self, x, t, meta: dict[str, torch.Tensor], cfg: float = 1.0):
        return self._guided_prediction(x, t, meta, cfg)

    def _guided_prediction(
        self,
        x,
        t,
        meta: dict[str, torch.Tensor],
        cfg: float = 1.0,
    ) -> torch.Tensor:  # fmt: off
        if t.ndim == 0:  # the ODE solver gives scalar t
            t = t.expand(x.shape[0])

        if cfg == 0.0:  # unconditional
            force_drop = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
            return self.model(x=x, t=t, meta=meta, force_drop_rec_conc=force_drop)
        if cfg == 1.0:  # conditional
            return self.model(x=x, t=t, meta=meta, force_drop_rec_conc=None)

        pred_cond = self.model(x=x, t=t, meta=meta, force_drop_rec_conc=None)
        force_drop = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        pred_null = self.model(x=x, t=t, meta=meta, force_drop_rec_conc=force_drop)
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
            return self._guided_prediction(x=x, t=t, meta=meta)

        traj = torchdiffeq.odeint(
            forward_fn,
            x0,
            torch.linspace(0, 1, 2, device=x0.device),
            method="dopri5",
            rtol=1e-5,
            atol=1e-5,
        )
        return traj[-1], nfe  # type: ignore

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def compute_loss(
        self,
        x0: torch.Tensor,  # shape (B, C, H, W), sampled from source distribution
        x1: torch.Tensor,  # shape (B, C, H, W), sampled from target distribution
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=torch.float32)  # (B,) - [0,1)
        t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

        xt = torch.lerp(x0, x1, t_)
        ut = x1 - x0

        vt = self(x=xt, t=t, meta=meta)
        return torch.nn.functional.mse_loss(vt, ut)
