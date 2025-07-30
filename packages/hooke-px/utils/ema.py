import copy
from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# v. rough guide to decay values based on planned training time:
# ~10k steps: 0.99
# ~100k steps: 0.999
# ~1M steps: 0.9999


class EMA(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, decay: float = 0.9999):
        super().__init__()
        self.module = copy.deepcopy(net)
        self.module.eval()
        self.decay = decay

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        model_device = self._get_model_device(model)
        ema_device = self._get_model_device(self.module)

        if model_device != ema_device:
            self.module.to(model_device)

        update_ema(self.module, model, decay=self.decay)

    def switch(self, model: torch.nn.Module):
        """Copy the EMA model's weights to the given model.
        Useful for the switch-EMA trick from [1].
        [1] arxiv.org/abs/2402.09240"""
        update_ema(model, self.module, decay=0.0)  # a bit hacky

    def sync(self, model: torch.nn.Module):
        """Copy the given model's weights to the EMA model.
        Useful for re-syncing the EMA model after a warmup period."""
        update_ema(self.module, model, decay=0.0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
