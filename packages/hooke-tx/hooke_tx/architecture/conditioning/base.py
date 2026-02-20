from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hooke_tx.architecture.conditioning.embedders import Identity, Fourier, OneHotEmbedder
from hooke_tx.architecture.modules.mlp import MLPBlock

from hooke_tx.data.constants import GENE_PERT, MOL_PERT, DOSE, EMPTY, ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL


EMBEDDER_DICT = {
    "identity": Identity,
    "fourier": Fourier,
    "one-hot": OneHotEmbedder,
}


class BaseConditioning(nn.Module):
    """
    Base class for conditioning models.
    """
    def __init__(
        self,
        covariates: dict[str, list[str | float]],
        embedding_args: dict[str, dict[str, Any]],
    ):
        super().__init__()
        # Remove covariates that only appear once
        covariates = {k: v for k, v in covariates.items() if len(v) > 1}

        # Define embedders for covariates; map config keys to internal names
        self.embedder_dict = nn.ModuleDict()
        for input_type, args in list(embedding_args.items()):
            if args is None:
                continue
            
            embedder_type = args.get("type")
            
            if input_type not in covariates and input_type not in ("time", "xt"):
                continue
            
            cov_list = covariates.get(input_type, [])
            self.embedder_dict[input_type] = self._parse_embedder_args(embedder_type, cov_list, args)


    def _parse_embedder_args(
        self,
        embedder_type: str,
        cov_list: list,
        args: dict[str, Any],
    ) -> nn.Module:
        """Build one embedder; only pass kwargs the class accepts."""
        if embedder_type == "identity":
            return Identity()
        elif embedder_type == "fourier":
            return Fourier(dim=args.get("dim"), bandwidth=args.get("bandwidth", 1))
        elif embedder_type == "one-hot":
            return OneHotEmbedder(labels=cov_list, dim=args.get("dim"))
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
        return embedder

    def forward(self, pre_embedding_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding_dict = {}
        for input_name in ["time", "xt", ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL]:
            if input_name in self.embedder_dict:
                embedding_dict[input_name] = self.embedder_dict[input_name].forward(pre_embedding_dict[input_name])
                
        # batch perturbations of different length
        batched_gene_perts = []
        gene_lengths = []
        for perts in pre_embedding_dict[GENE_PERT]:
            length = len(perts)
            batched_gene_perts.extend([p["ensembl_gene_id"] for p in perts] + [EMPTY] * (3 - length))

        batched_mol_perts = []
        batched_mol_doses = []
        for perts in pre_embedding_dict[MOL_PERT]:
            length = len(perts)
            batched_mol_perts.extend([p["smiles"] for p in perts] + [EMPTY] * (3 - length))
            batched_mol_doses.extend([p["concentration"] for p in perts] + [EMPTY] * (3 - length))

        batched_gene_perts = torch.tensor(batched_gene_perts)
        batched_mol_perts = torch.tensor(batched_mol_perts)
        batched_mol_doses = torch.tensor(batched_mol_doses)
        
        # Embed
        embedded_batched_gene_perts = self.embedder_dict[GENE_PERT].forward(batched_gene_perts)
        embedded_batched_mol_perts = self.embedder_dict[MOL_PERT].forward(batched_mol_perts)
        embedded_batched_mol_doses = self.embedder_dict[DOSE].forward(batched_mol_doses)

        # Unbatch by using .view leveraging that all perturbations were padded to lenght 3
        embedding_dict[GENE_PERT] = embedded_batched_gene_perts.view(3, -1, embedded_batched_gene_perts.size(-1))
        embedding_dict[MOL_PERT] = embedded_batched_mol_perts.view(3, -1, embedded_batched_mol_perts.size(-1))
        embedding_dict[DOSE] = embedded_batched_mol_doses.view(3, -1, embedded_batched_mol_doses.size(-1))
        
        return embedding_dict


class ConditioningMLP(BaseConditioning):
    def __init__(
        self,
        covariates: dict[str, list[str | float]],
        embedding_args: dict[str, dict[str, Any]],
        data_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        pre_layers: int = 2,
        post_layers: int = 2,
    ):
        super().__init__(
            covariates=covariates,
            embedding_args=embedding_args,
        )

        self.proj_dict = nn.ModuleDict()
        embedding_args["xt"]["dim"] = data_dim
        for input_type, args in embedding_args.items():
            embedding_dim = args.get("dim")
            if hidden_dim != embedding_dim:
                self.proj_dict[input_type] = nn.Linear(embedding_dim, hidden_dim)

        mol_mlp_layers = [
            MLPBlock(
                in_dim=3*hidden_dim,
                out_dim=hidden_dim,
            ),
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
            )
        ]
        self.mol_mlp = nn.Sequential(*mol_mlp_layers)

        gene_mlp_layers = [
            MLPBlock(
                in_dim=3*hidden_dim,
                out_dim=hidden_dim,
            ),
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
            )
        ]
        self.gene_mlp = nn.Sequential(*gene_mlp_layers)

        mlp_xt_layers = [
            MLPBlock(
                in_dim=2*hidden_dim,
                out_dim=hidden_dim,
            )
        ]
        for _ in range(pre_layers - 2):
            mlp_xt_layers.append(
                MLPBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                )
            )

        mlp_xt_layers.append(
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=latent_dim,
            )
        )
        self.mlp_xt = nn.Sequential(*mlp_xt_layers)

        mlp_p_layers = [
            MLPBlock(
                in_dim=2*hidden_dim,
                out_dim=hidden_dim,
            )
        ]
        for _ in range(pre_layers - 2):
            mlp_p_layers.append(
                MLPBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                )
            )
        mlp_p_layers.append(
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=latent_dim,
            )
        )
        self.mlp_p = nn.Sequential(*mlp_p_layers)

        mlp_c_layers = [
            MLPBlock(
                in_dim=3*hidden_dim,
                out_dim=hidden_dim,
            )
        ]
        for _ in range(pre_layers - 2):
            mlp_c_layers.append(
                MLPBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                )
            )
        mlp_c_layers.append(
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=latent_dim,
            )
        )
        self.mlp_c = nn.Sequential(*mlp_c_layers)
        
        post_mlp_layers = [  # Initial dropout before first layer
            MLPBlock(
                in_dim=3*latent_dim,
                out_dim=hidden_dim,
            )
        ]
        for _ in range(post_layers - 2):
            post_mlp_layers.append(
                MLPBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                )
            )
        post_mlp_layers.append(
            MLPBlock(
                in_dim=hidden_dim,
                out_dim=data_dim,
            )
        )
        self.post_mlp = nn.Sequential(*post_mlp_layers)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        pre_embedding_dict = {"time": t, "xt": xt}
        pre_embedding_dict.update(covariates)
        embedding_dict = super().forward(pre_embedding_dict)

        # mol perturbations
        mol_dose_embeddings = embedding_dict[MOL_PERT] + embedding_dict[DOSE]
        flattened_mol_dose_embeddings = mol_dose_embeddings.view(mol_dose_embeddings.size(0), -1)
        flattened_gene_embeddings = embedding_dict[GENE_PERT].view(embedding_dict[GENE_PERT].size(0), -1)

        xt_combined = torch.cat(
            [
                embedding_dict["xt"],
                embedding_dict["time"],
            ],
            dim=-1,
        )

        p_combined = torch.cat(
            [
                embedding_dict[ROUTING],
                self.mol_mlp(flattened_mol_dose_embeddings),
                self.gene_mlp(flattened_gene_embeddings),
            ],
            dim=-1,
        )
        
        c_combined = torch.cat(
            [
                embedding_dict[CELL_TYPE],
                embedding_dict[EXPERIMENT],
                embedding_dict[WELL],
            ],
            dim=-1,
        )
        
        all_combined = torch.cat(
            [
                self.mlp_xt(xt_combined),
                self.mlp_p(p_combined),
                self.mlp_c(c_combined),
            ],
            dim=-1,
        )
        
        conditioning = self.post_mlp(all_combined)

        return conditioning