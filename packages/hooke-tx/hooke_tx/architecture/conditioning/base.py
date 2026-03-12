from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hooke_tx.architecture.conditioning.embedders import (
    Identity,
    Projection,
    Fourier,
    OneHotEmbedder,
    ECFPEmbedder,
    MolGPSEmbedder,
)
from hooke_tx.architecture.modules.mlp import MLPBlock

from hooke_tx.data.constants import GENE_PERT, MOL_PERT, DOSE, EMPTY, ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_middle: int = 0) -> nn.Sequential:
    layers = [MLPBlock(in_dim, hidden_dim)]
    
    for _ in range(n_middle):
        layers.append(MLPBlock(hidden_dim, hidden_dim))
    
    layers.append(MLPBlock(hidden_dim, out_dim))
    
    return nn.Sequential(*layers)


def _embedder_dim(emb: nn.Module, input_type: str, data_dim: int) -> int:
    if isinstance(emb, Identity):
        return data_dim if input_type == "xt" else 1
    
    if hasattr(emb, "embedding"):
        return emb.embedding.weight.shape[-1]
    
    if hasattr(emb, "fingerprint_matrix"):
        return emb.fingerprint_matrix.shape[-1]
    
    if hasattr(emb, "proj"):
        return emb.proj.out_features
    
    if hasattr(emb, "freqs"):
        return emb.freqs.shape[0]
    
    return 0


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

        for input_type, args_list in list(embedding_args.items()):
            if args_list is None:
                continue

            if input_type not in covariates and input_type not in ("time", "xt"):
                continue

            cov_list = covariates.get(input_type, [])
            
            embedders = nn.ModuleDict({
                f"{args.get('type', 'one-hot')}_{i}": self._create_embedder(args, cov_list, data_dim)
                for i, args in enumerate(args_list)
            })
            
            self.embedder_dict[input_type] = embedders
            
            dims = [_embedder_dim(m, input_type, data_dim) for m in embedders.values()]
            self._output_dims[input_type] = sum(dims)

    def _create_embedder(
        self,
        args: dict[str, Any],
        cov_list: list,
        data_dim: int,
    ) -> nn.Module:
        embedder_type = args.get("type", "one-hot")
        
        if embedder_type == "identity":
            return Identity()
        
        if embedder_type == "projection":
            return Projection(in_dim=data_dim, dim=args.get("dim"))
        
        if embedder_type == "fourier":
            return Fourier(dim=args.get("dim"), bandwidth=args.get("bandwidth", 1))
        
        if embedder_type == "one-hot":
            return OneHotEmbedder(all_labels=cov_list, dim=args.get("dim"))
        
        if embedder_type == "ecfp":
            return ECFPEmbedder(
                all_labels=cov_list,
                dim=args.get("dim", 1024),
                radius=args.get("radius", 2),
            )
        
        if embedder_type == "molgps":
            emb_name = args.get("emb_name")
            
            return MolGPSEmbedder(
                all_labels=cov_list,
                emb_name=emb_name,
                cache_dir=args.get("molgps_cache_dir", "/rxrx/data/valence/pef/molgps"),
            )
        
        raise ValueError(f"Unknown embedder type: {embedder_type}")

    def get_output_dims(self) -> dict[str, int]:
        return dict(self._output_dims)

    def _call_embedder(self, embedder: nn.ModuleDict, data) -> torch.Tensor:
        return torch.cat([m(data) for m in embedder.values()], dim=-1)

    def forward(self, pre_embedding_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding_dict = {}
        for input_name in ["time", "xt", ROUTING, ASSAY, CELL_TYPE, EXPERIMENT, WELL]:
            if input_name in self.embedder_dict:
                embedding_dict[input_name] = self._call_embedder(
                    self.embedder_dict[input_name],
                    pre_embedding_dict[input_name],
                )

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

        embedded_gene_perts = self._call_embedder(self.embedder_dict[GENE_PERT], batched_gene_perts)
        embedded_mol_perts = self._call_embedder(self.embedder_dict[MOL_PERT], batched_mol_perts)
        embedded_mol_doses = self._call_embedder(self.embedder_dict[DOSE], batched_mol_doses)

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

        self.mol_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.gene_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.mlp_xt = _mlp(2 * hidden_dim, hidden_dim, latent_dim, pre_layers - 2)
        self.mlp_p = _mlp(3 * hidden_dim, hidden_dim, latent_dim, pre_layers - 2)
        self.mlp_c = _mlp(3 * hidden_dim, hidden_dim, latent_dim, pre_layers - 2)
        self.post_mlp = _mlp(3 * latent_dim, hidden_dim, data_dim, post_layers - 2)

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