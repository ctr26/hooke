from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hooke_tx.architecture.conditioning.embedders import Identity, Projection, Fourier, OneHotEmbedder
from hooke_tx.architecture.modules.mlp import MLPBlock

from hooke_tx.data.constants import GENE_PERT, MOL_PERT, DOSE, EMPTY, ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL


EMBEDDER_DICT = {
    "identity": Identity,
    "projection": Projection,
    "fourier": Fourier,
    "one-hot": OneHotEmbedder,
}


def _embedder_output_dim(
    embedder_type: str,
    input_type: str,
    data_dim: int,
    args: dict[str, Any],
) -> int:
    """Output feature dim for an embedder (for building proj_dict)."""
    if embedder_type == "identity":
        return data_dim if input_type == "xt" else 1
    
    return args.get("dim")


class EmbeddingModule(nn.Module):
    """
    Embeds raw inputs (time, xt, covariates) into a dict of vectors.
    Output dims can vary per key; use ConditioningModule's proj step to align.
    """
    def __init__(
        self,
        data_dim: int,
        covariates: dict[str, list[str | float]],
        embedding_args: dict[str, dict[str, Any]],
    ):
        super().__init__()
        self.embedder_dict = nn.ModuleDict()
        self._output_dims: dict[str, int] = {}

        for input_type, args in list(embedding_args.items()):
            if args is None:
                continue

            embedder_type = args.get("type")
            if input_type not in covariates and input_type not in ("time", "xt"):
                continue

            cov_list = covariates.get(input_type, [])
            self.embedder_dict[input_type] = self._parse_embedder_args(
                embedder_type, cov_list, data_dim, args, input_type
            )
            self._output_dims[input_type] = _embedder_output_dim(
                embedder_type, input_type, data_dim, args
            )

    def _parse_embedder_args(
        self,
        embedder_type: str,
        cov_list: list,
        data_dim: int,
        args: dict[str, Any],
        input_type: str,
    ) -> nn.Module:
        if embedder_type == "identity":
            return Identity()
        elif embedder_type == "projection":
            return Projection(in_dim=data_dim, dim=args.get("dim"))
        elif embedder_type == "fourier":
            return Fourier(dim=args.get("dim"), bandwidth=args.get("bandwidth", 1))
        elif embedder_type == "one-hot":
            return OneHotEmbedder(labels=cov_list, dim=args.get("dim"))
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    def get_output_dims(self) -> dict[str, int]:
        return dict(self._output_dims)

    def forward(self, pre_embedding_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding_dict = {}
        for input_name in ["time", "xt", ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL]:
            if input_name in self.embedder_dict:
                embedding_dict[input_name] = self.embedder_dict[input_name](pre_embedding_dict[input_name])

        batched_gene_perts = []
        for perts in pre_embedding_dict[GENE_PERT]:
            length = len(perts)
            batched_gene_perts.extend([p[1] for p in perts] + [EMPTY] * (3 - length))

        batched_mol_perts = []
        batched_mol_doses = []
        for perts in pre_embedding_dict[MOL_PERT]:
            length = len(perts)
            batched_mol_perts.extend([p[1] for p in perts] + [EMPTY] * (3 - length))
            batched_mol_doses.extend([p[2] for p in perts] + [EMPTY] * (3 - length))

        embedded_gene_perts = self.embedder_dict[GENE_PERT](batched_gene_perts)
        embedded_mol_perts = self.embedder_dict[MOL_PERT](batched_mol_perts)
        embedded_mol_doses = self.embedder_dict[DOSE](batched_mol_doses)

        embedding_dict[GENE_PERT] = embedded_gene_perts.view(3, -1, embedded_gene_perts.size(-1))
        embedding_dict[MOL_PERT] = embedded_mol_perts.view(3, -1, embedded_mol_perts.size(-1))
        embedding_dict[DOSE] = embedded_mol_doses.view(3, -1, embedded_mol_doses.size(-1))
        
        return embedding_dict


class ConditioningMLP(nn.Module):
    """
    Conditioning module: takes embedding_dict, projects to hidden_dim, then fuses via MLPs.
    Uses EmbeddingModule for the embedding step; proj_dict is the first step here.
    """
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
        super().__init__()
        self.embedder = EmbeddingModule(
            data_dim=data_dim,
            covariates=covariates,
            embedding_args=embedding_args,
        )
        output_dims = self.embedder.get_output_dims()

        self.proj_dict = nn.ModuleDict()
        for input_type, embedding_dim in output_dims.items():
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
                in_dim=3*hidden_dim,
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

    def _project_embeddings(self, embedding_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """First step of conditioning: align all embedding dims to hidden_dim."""
        out = {}
        for k, v in embedding_dict.items():
            if k in self.proj_dict:
                out[k] = self.proj_dict[k](v)
            else:
                out[k] = v
        return out

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        pre_embedding_dict = {"time": t, "xt": xt}
        pre_embedding_dict.update(covariates)
        embedding_dict = self.embedder(pre_embedding_dict)
        embedding_dict = self._project_embeddings(embedding_dict)

        # mol perturbations
        mol_dose_embeddings = embedding_dict[MOL_PERT] + embedding_dict[DOSE]
        flattened_mol_dose_embeddings = mol_dose_embeddings.permute(1, 0, 2).reshape(mol_dose_embeddings.size(1), -1)
        flattened_gene_embeddings = embedding_dict[GENE_PERT].permute(1, 0, 2).reshape(embedding_dict[GENE_PERT].size(1), -1)

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