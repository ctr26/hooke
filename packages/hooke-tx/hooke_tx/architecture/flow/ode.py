from torchdiffeq import odeint
import torch


class ODESolver:
    def __init__(self, model, rtol=1e-5, atol=1e-5, method="dopri5"):
        self.model = model
        self.odeint = lambda x, t, ode_opts: odeint(
            func=self.model,
            y0=x,
            t=t,
            method=method,
            options=ode_opts,
            rtol=rtol,
            atol=atol,
        )

    def sample(
        self, x_init, step_size=None, time_grid: torch.Tensor = torch.tensor([0.0, 1.0])
    ):
        ode_opts = {"step_size": step_size} if step_size is not None else {}
        return self.odeint(x=x_init, t=time_grid.to(x_init.device), ode_opts=ode_opts)