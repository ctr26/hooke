import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from hooke_tx.architecture.modules.mlp import MLPBlock
from hooke_tx.architecture.modules.emb import MPFourier

from hooke_tx.data.chem_utils import compute_ecfp_embeddings, retrieve_px_pretrained_embeddings, retrieve_molgps_embeddings

from hooke_tx.data.constants import BASE_LABEL, CONTROL_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL


class BasicMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        mol_perts: list[str],
        mol_pert_names: list[str],
        doses: list[str],
        gene_perts: list[str],
        cell_types: list[str],
        batches: list[str] = None,
        embedding_dim_universal: int = None,
        embedding_dim_m: int = 128,
        embedding_dim_d: int = 128,
        embedding_dim_g: int = 128,
        embedding_dim_c: int = 128,
        embedding_dim_b: int = None,
        embedding_dim_t: int = 32,
        proj_nn_config: dict[str, str] = {},
        hidden_dim: int = 256,
        latent_dim: int = 256,
        c_layers: int = 2,
        ut_layers: int = 2,
        norm: bool = False,
        dropout: float = 0.0,
        concat_latent: bool = True,
        skip_type: str = "none",  # "none", "sum", or "cat"
        dim_multiplier: int = 1,
        embed_mol_pert: str = "one-hot",
        embed_dose: str = "one-hot",
        condition_on_dose: bool = True,
        condition_on_batch: bool = False,
        condition_on_expressed_genes: bool = False,
    ):
        super().__init__()
        assert c_layers >= 2, "c_layers must be at least 2"
        assert ut_layers >= 2, "ut_layers must be at least 2"
        assert concat_latent in [True, False], "concat_latent must be a boolean"
        assert skip_type in ["none", "sum", "cat"], \
            f"skip_type must be 'none', 'sum', or 'cat', got {skip_type}"

        doses = list(set([float(d) for d in doses]))
        mol_perts = [m for m in mol_perts if not pd.isna(m) and m not in [BASE_LABEL, CONTROL_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL]]
        mol_pert_names = [m for m in mol_pert_names if m not in [BASE_LABEL, CONTROL_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL]]

        if condition_on_batch:
            assert batches is not None, "batches must be provided if condition_on_batch is True"

        if embed_mol_pert == "one-hot":
            proj_nn_config.pop("m", None)

        if embed_dose == "one-hot":
            proj_nn_config.pop("d", None)

        if embedding_dim_universal is not None:
            embedding_dim_m = embedding_dim_universal
            embedding_dim_d = embedding_dim_universal
            embedding_dim_g = embedding_dim_universal
            embedding_dim_c = embedding_dim_universal
            embedding_dim_b = embedding_dim_universal
            embedding_dim_t = embedding_dim_universal
        
        hidden_dim = hidden_dim * dim_multiplier
        latent_dim = latent_dim * dim_multiplier

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_mol_pert = embed_mol_pert
        self.concat_latent = concat_latent
        self.condition_on_dose = condition_on_dose
        self.condition_on_batch = condition_on_batch
        self.condition_on_expressed_genes = condition_on_expressed_genes
        
        self.t_embeddings = MPFourier(num_channels=embedding_dim_t)

        if embed_mol_pert == "one-hot":
            mol_perts = mol_perts + [BASE_LABEL, CONTROL_LABEL, NEG_CONTROL_LABEL, UNKNOWN_LABEL]
            self.m_embeddings = nn.Embedding(len(mol_perts), embedding_dim_m)
        
        elif embed_mol_pert == "ecfp":
            embedding_dim_m = 1024
            mol_perts, embedding_dim_m, ecfp_matrix = compute_ecfp_embeddings(mol_perts, embedding_dim_m)
            self.m_embeddings = nn.Embedding.from_pretrained(
                ecfp_matrix, freeze=True, padding_idx=None
            )
        elif embed_mol_pert == "px_pretrained":
            mol_perts, embedding_dim_m, embedding_matrix = retrieve_px_pretrained_embeddings(mol_pert_names)
            self.m_embeddings = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=True, padding_idx=None
            )
        elif isinstance(embed_mol_pert, list):
            mol_perts, embedding_dim_m, molgps_matrix = retrieve_molgps_embeddings(mol_perts, embed_mol_pert)
            self.m_embeddings = nn.Embedding.from_pretrained(
                molgps_matrix, freeze=True, padding_idx=None
            )
        else:
            raise ValueError(f"Invalid embedding type: {embed_mol_pert}")

        if embed_dose == "one-hot":
            self.d_embeddings = nn.Embedding(len(doses), embedding_dim_d)
        elif embed_dose == "numeric":
            embedding_dim_d = 2
            self.d_embeddings = nn.Embedding.from_pretrained(
                torch.tensor([[float(d), np.log1p(float(d))] for d in doses], dtype=torch.float32), freeze=True, padding_idx=None
            )
        elif embed_dose == "polynomial":
            raise NotImplementedError("Polynomial dose embedding is not implemented yet")
        else:
            raise ValueError(f"Invalid embedding type: {embed_dose}")

        raw_embedding_dim_dict = {
            "m": embedding_dim_m,
            "d": embedding_dim_d,
            "g": embedding_dim_g,
            "c": embedding_dim_c,
            "b": embedding_dim_b,
            "t": embedding_dim_t,
            "xt": data_dim,
            "xte": data_dim,
        }
        embedding_dim_dict = raw_embedding_dim_dict.copy()

        self.g_embeddings = nn.Embedding(len(gene_perts), embedding_dim_g)
        self.c_embeddings = nn.Embedding(len(cell_types), embedding_dim_c)
        self.b_embeddings = nn.Embedding(len(batches), embedding_dim_b) if condition_on_batch else None

        self.m2id = {pert: i for i, pert in enumerate(mol_perts)}
        self.d2id = {dose: i for i, dose in enumerate(doses)}
        self.g2id = {pert: i for i, pert in enumerate(gene_perts)}
        self.c2id = {context: i for i, context in enumerate(cell_types)}
        self.b2id = {batch: i for i, batch in enumerate(batches)} if condition_on_batch else None

        self.proj_nn_dict = nn.ModuleDict()
        for emb_name, dim in proj_nn_config.items():
            out_dim = dim if embedding_dim_universal is None else embedding_dim_universal
            self.proj_nn_dict[emb_name] = nn.Linear(raw_embedding_dim_dict[emb_name], out_dim)
            embedding_dim_dict[emb_name] = out_dim

        in_dim_c = embedding_dim_dict["m"] + embedding_dim_dict["g"] + embedding_dim_dict["c"]
        if condition_on_dose:
            in_dim_c += embedding_dim_dict["d"]
        if condition_on_batch:
            in_dim_c += embedding_dim_dict["b"]

        mlp_xt_layers = [
            MLPBlock(
                embedding_dim_dict["xt"] + embedding_dim_dict["t"] if not condition_on_expressed_genes else embedding_dim_dict["xt"] * 2 + embedding_dim_dict["t"],
                hidden_dim,
                norm=norm,
                activation=nn.GELU(),
                dropout=dropout,
                skip_type="none",  # First layer, no skip
            )
        ]
        for _ in range(c_layers - 2):
            mlp_xt_layers.append(
                MLPBlock(
                    hidden_dim,
                    hidden_dim,
                    norm=norm,
                    activation=nn.GELU(),
                    dropout=dropout,
                    skip_type=skip_type,
                )
            )
        mlp_xt_layers.append(
            MLPBlock(
                hidden_dim,
                latent_dim,
                norm=norm,
                activation=nn.GELU(),
                dropout=0.0,  # No dropout on last layer
                skip_type="none",  # Last layer, no skip
            )
        )
        self.mlp_xt = nn.Sequential(*mlp_xt_layers)

        mlp_c_layers = [
            MLPBlock(
                in_dim_c,
                hidden_dim,
                norm=norm,
                activation=nn.GELU(),
                dropout=dropout,
                skip_type="none",  # First layer, no skip
            )
        ]
        for _ in range(c_layers - 2):
            mlp_c_layers.append(
                MLPBlock(
                    hidden_dim,
                    hidden_dim,
                    norm=norm,
                    activation=nn.GELU(),
                    dropout=dropout,
                    skip_type=skip_type,
                )
            )
        mlp_c_layers.append(
            MLPBlock(
                hidden_dim,
                latent_dim,
                norm=norm,
                activation=nn.GELU(),
                dropout=0.0,  # No dropout on last layer
                skip_type="none",  # Last layer, no skip
            )
        )
        self.mlp_c = nn.Sequential(*mlp_c_layers)

        ut_input_dim = latent_dim if not concat_latent else 2 * latent_dim
        mlp_ut_layers = [
            nn.Dropout(dropout),  # Initial dropout before first layer
            MLPBlock(
                ut_input_dim,
                hidden_dim,
                norm=norm,
                activation=nn.GELU(),
                dropout=dropout,
                skip_type="none",  # First layer, no skip
            )
        ]
        for _ in range(ut_layers - 2):
            mlp_ut_layers.append(
                MLPBlock(
                    hidden_dim,
                    hidden_dim,
                    norm=norm,
                    activation=nn.GELU(),
                    dropout=dropout,
                    skip_type=skip_type,
                )
            )
        mlp_ut_layers.append(
            MLPBlock(
                hidden_dim,
                data_dim,
                norm=None,  # No normalization on output layer for regression
                activation=None,  # No activation on last layer
                dropout=0.0,  # No dropout on last layer
                skip_type="none",  # Last layer, no skip
            )
        )
        self.mlp_ut = nn.Sequential(*mlp_ut_layers)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        m_list: list[str],
        d_list: list[str],
        g_list: list[str],
        c_list: list[str],
        b_list: list[str] = None,
        m_name_list: list[str] = None,
        uncond: bool = False,
    ):
        m_list = m_name_list if self.embed_condition == "px_pretrained" else m_list

        emb_dict = {
            "m": self.m_embeddings(torch.tensor([self.m2id[m] for m in m_list], device=self.device)),
            "d": self.d_embeddings(torch.tensor([self.dose2id[d] for d in d_list], device=self.device)),
            "g": self.g_embeddings_embeddings(torch.tensor([self.g2id[g] for g in g_list], device=self.device)),
            "c": self.c_embeddings(torch.tensor([self.c2id[c] for c in c_list], device=self.device)),
            "b": self.b_embeddings(torch.tensor([self.b2id[b] for b in b_list], device=self.device)) if self.condition_on_batch else None,
            "t": self.t_embeddings(t),
            "xt": xt,
            "xte": (xt > 1e-12).to(dtype=torch.float32, device=self.device) if self.condition_on_expressed_genes else None,
        }

        # Zero out condition embeddings for unconditional pass (CFG)
        if uncond:
            emb_dict["p"] = torch.zeros_like(emb_dict["p"])
            emb_dict["d"] = torch.zeros_like(emb_dict["d"])
            emb_dict["g"] = torch.zeros_like(emb_dict["g"])
            emb_dict["c"] = torch.zeros_like(emb_dict["c"])
            if emb_dict["b"] is not None:
                emb_dict["b"] = torch.zeros_like(emb_dict["b"])

        for emb_name, proj in self.proj_nn_dict.items():
            if emb_dict.get(emb_name, None) is not None:
                projected = proj(emb_dict[emb_name])
                emb_dict[emb_name] = nn.GELU()(projected)

        xt_emb_list = [emb_dict["xt"], emb_dict["t"]]
        if self.condition_on_expressed_genes:
            xt_emb_list.append(emb_dict["xte"])

        xt_emb = torch.cat(xt_emb_list, dim=-1)

        c_emb_list = [emb_dict["m"]]
        if self.condition_on_batch:
            c_emb_list.append(emb_dict["b"])
        c_emb_list.append(emb_dict["g"])
        c_emb_list.append(emb_dict["c"])
        if self.condition_on_dose:
            c_emb_list.append(emb_dict["d"])

        c_emb = torch.cat(c_emb_list, dim=-1)

        xt_emb = self.mlp_xt(xt_emb)
        c_emb = self.mlp_c(c_emb)

        combined_latent = torch.cat([xt_emb, c_emb], dim=-1) if self.concat_latent else xt_emb + c_emb

        conditioning = self.mlp_ut(combined_latent)

        return conditioning