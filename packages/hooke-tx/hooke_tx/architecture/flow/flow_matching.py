from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hooke_tx.architecture.flow.ode import ODESolver


def predict_with_cfg(
    conditioning_model: nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    covariates: dict[str, torch.Tensor],
    guidance_strength: float = 1.0,
) -> torch.Tensor:
    if t.ndim == 0:  # the solvers from the flow_matching lib give scalar t
        t = t.expand(xt.shape[0])

    if guidance_strength == 1:  # pure conditional, skip uncond pass
        vt = conditioning_model(xt, t, covariates)
    elif guidance_strength == 0:  # pure unconditional, skip cond pass
        vt = conditioning_model(xt, t, covariates, uncond=True)
    else:
        cond = conditioning_model(xt, t, covariates)
        uncond = conditioning_model(xt, t, covariates, uncond=True)
        vt = uncond + guidance_strength * (cond - uncond)
    
    return vt


class FlowMatching(nn.Module):
    """
    A modern flow matching architecture for Tx data.
    """
    def __init__(
        self,
        conditioning_model: nn.Module,
        data_dim: int,
        flow: str = "D2D",
        guidance_strength: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.conditioning_model = conditioning_model
        self.data_dim = data_dim
        self.mode = flow
        self.guidance_strength = guidance_strength

    def compute_loss(
        self,
        x0: torch.Tensor,  # shape (B, D)
        x1: torch.Tensor,  # shape (B, D)
        covariates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the flow matching (linear interpolant) loss."""
        t = torch.rand(
            x0.shape[0], dtype=torch.float32, device=x0.device
        )  # shape (B,) - [0,1)
        t_expanded = t.reshape(-1, 1).expand_as(x0)  # (B,) -> (B, D)

        xt = torch.lerp(x0, x1, t_expanded)
        ut = x1 - x0

        ut_pred = self.conditioning_model(xt, t, covariates)
        
        return F.mse_loss(ut_pred, ut)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor = None,
        covariates: dict[str, torch.Tensor] = None,
    ):
        """Forward with flexible kwargs to satisfy BaseModel interface."""
        if self.mode == "N2D":
            x0 = torch.randn_like(x1)
        
        loss = self.compute_loss(x0, x1, covariates)

        if t is not None:
            xt = torch.lerp(x0, x1, t.reshape(-1, 1).expand_as(x0))
            x = self.conditioning_model(xt, t, covariates)
            return loss, x
        return loss

    def generate(
        self,
        xt: torch.Tensor,  # from base distribution, shape (B, N)
        covariates: dict[str, torch.Tensor],
        guidance_strength: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Generating a sample by solving the probability flow ODE."""
        self.eval()  # type: ignore[attr-defined]
        nfe = 0

        if self.mode == "N2D":
            xt = torch.randn(xt.size(0), self.data_dim, device=xt.device)

        # solver assumes that forward_fn is a function of x and t only
        def forward_fn(t, x):
            nonlocal nfe  # to make closures work properly in python
            nfe += 1
            return predict_with_cfg(
                conditioning_model=self.conditioning_model,
                t=t,
                xt=x,
                covariates=covariates,
                guidance_strength=guidance_strength,
            )

        solver = ODESolver(forward_fn)
        solution = solver.sample(
            x_init=xt,
            step_size=None,
        )
        assert isinstance(solution, torch.Tensor)
        return solution, nfe