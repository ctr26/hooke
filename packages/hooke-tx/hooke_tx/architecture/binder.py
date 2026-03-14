from typing import Any

import torch
import torch.nn as nn


from hooke_tx.architecture.conditioning.base import ConditioningMLP
from hooke_tx.architecture.flow.flow_matching import FlowMatching

CONDITION_MODEL_DICT = {
    "mlp": ConditioningMLP,
}
GEN_MODEL_DICT = {
    "flow": FlowMatching,
}


class TxGenBinder(nn.Module):
    """
    Variable archtecture for Tx generation.
    """
    def __init__(
        self,
        covariates: dict[str, list[str | float]],
        data_dim: int,
        embedding_args: dict[str, dict[str, str]],
        conditioning_args: dict[str, Any],
        generation_args: dict[str, Any],
    ):
        super().__init__()

        conditioning_type = conditioning_args.pop("type")
        generation_type = generation_args.pop("type")

        conditioning_model = CONDITION_MODEL_DICT[conditioning_type](
            covariates=covariates,
            data_dim=data_dim,
            embedding_args=embedding_args,
            **conditioning_args
        )
        self.generation_model = GEN_MODEL_DICT[generation_type](
            conditioning_model=conditioning_model,
            data_dim=data_dim,
            **generation_args
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        x0 = batch["x0"]
        x1 = batch["x1"]
        covariates = batch["covariates"]
        return self.generation_model.compute_loss(x0, x1, covariates)

    def generate(self, batch: dict[str, Any]) -> torch.Tensor:
        xt = batch.get("xt", batch["x0"])
        covariates = batch["covariates"]
        pred, _ = self.generation_model.generate(xt, covariates)
        return pred