"""Conditioning layers for the DiT generator."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
from layers import TransformerCore

import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


class ScalarEmbedder(nn.Module):
    """Embeds scalar conditions into vector representations. Useful for timesteps, zoom levels, etc."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations. Also handles dropout for cfg."""

    def __init__(self, num_classes, hidden_size, dropout_prob=0.15):
        super().__init__()
        # +1 for null token
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        # Handle 2D labels (e.g., sequence of tokens) by adding broadcast dimension
        if labels.dim() > 1:
            drop_ids = drop_ids.unsqueeze(-1)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class BasicAdaptor(nn.Module):
    """Adaptor module for conditioning layers in DiT generator.
    Uses OH embeddings (useful for pretraining but cannot generalise to new conditions).
    Assumes the following conditioning:
    - t: a scalar time condition, float range [0,1]
    - z: a scalar zoom condition, float range [0,1]
    - y: a class label, int range [0, y_dim-1]
    - e: an experiment label, int range [0, e_dim-1]
    - c: a cell type label, int range [0, c_dim-1]
    """

    def __init__(self, hidden_size, y_dim, e_dim, c_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embedder = ScalarEmbedder(hidden_size, frequency_embedding_size)
        self.z_embedder = ScalarEmbedder(hidden_size, frequency_embedding_size)
        self.y_embedder = LabelEmbedder(num_classes=y_dim, hidden_size=hidden_size)
        self.e_embedder = LabelEmbedder(num_classes=e_dim, hidden_size=hidden_size)
        self.c_embedder = LabelEmbedder(num_classes=c_dim, hidden_size=hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)  # type: ignore

        nn.init.normal_(self.z_embedder.mlp[0].weight, std=0.02)  # type: ignore
        nn.init.normal_(self.z_embedder.mlp[2].weight, std=0.02)  # type: ignore

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.e_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.c_embedder.embedding_table.weight, std=0.02)

    def forward(self, t, z, y, e, c):
        t_emb = self.t_embedder(t)
        z_emb = self.z_embedder(z)

        # allow passing None to use unconditional embedding for cfg
        if y is None:
            y = torch.full((t.shape[0],), self.y_embedder.num_classes, device=t.device)
        if e is None:
            e = torch.full((t.shape[0],), self.e_embedder.num_classes, device=t.device)
        if c is None:
            c = torch.full((t.shape[0],), self.c_embedder.num_classes, device=t.device)

        y_emb = self.y_embedder(y, train=self.training)
        e_emb = self.e_embedder(e, train=self.training)
        c_emb = self.c_embedder(c, train=self.training)
        return t_emb + z_emb + y_emb + e_emb + c_emb


class TransformerAdaptor(nn.Module):
    def __init__(
        self,
        hidden_size,
        rec_id_dim,
        concentration_dim,
        cell_type_dim,
        experiment_dim,
        image_type_dim,
        well_address_dim,
        dropout_prob=0.15,
    ):
        super().__init__()
        self.rec_id_embedder = LabelEmbedder(
            num_classes=rec_id_dim, hidden_size=hidden_size, dropout_prob=dropout_prob
        )
        self.concentration_embedder = LabelEmbedder(
            num_classes=concentration_dim,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
        )
        self.cell_type_embedder = LabelEmbedder(
            num_classes=cell_type_dim,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
        )
        self.experiment_embedder = LabelEmbedder(
            num_classes=experiment_dim,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
        )
        self.image_type_embedder = LabelEmbedder(
            num_classes=image_type_dim, hidden_size=hidden_size, dropout_prob=0.0
        )
        self.well_address_embedder = LabelEmbedder(
            num_classes=well_address_dim,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
        )

        self.transformer = TransformerCore(
            n_embd=hidden_size, n_layer=8, n_head=8, dropout=0.0
        )
        self.class_token = nn.Parameter(
            torch.randn(1, 1, hidden_size) * 0.02, requires_grad=True
        )
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.rec_id_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.concentration_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.cell_type_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.experiment_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.image_type_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.well_address_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)

    def forward(
        self,
        rec_id,
        concentration,
        cell_type,
        experiment_label,
        image_type,
        well_address,
        comp_mask=None,
        force_drop_rec_conc: torch.Tensor | None = None,
    ):
        if comp_mask is None:
            comp_mask = torch.ones_like(rec_id, dtype=torch.bool)

        # drop whole sequences (per-sample) for CFG training/inference
        rec_id_emb = self.rec_id_embedder(
            rec_id, train=self.training, force_drop_ids=force_drop_rec_conc
        )
        concentration_emb = self.concentration_embedder(
            concentration, train=self.training, force_drop_ids=force_drop_rec_conc
        )
        transformer_input = torch.cat(
            [
                rec_id_emb + concentration_emb,
                self.cell_type_embedder(cell_type, train=self.training).unsqueeze(1),
                self.experiment_embedder(
                    experiment_label, train=self.training
                ).unsqueeze(1),
                self.image_type_embedder(image_type, train=self.training).unsqueeze(1),
                self.well_address_embedder(well_address, train=self.training).unsqueeze(
                    1
                ),
                self.class_token.expand(rec_id.shape[0], -1, -1),
            ],
            dim=1,
        )
        keep = torch.cat(
            [
                comp_mask,
                torch.ones(
                    (rec_id.shape[0], transformer_input.shape[1] - rec_id.shape[1]),
                    device=rec_id.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )

        transformer_output = self.transformer(transformer_input, mask=keep)
        return transformer_output[:, -1, :]


class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = []

    def fit(self, df: pl.DataFrame):
        unique_tokens = df.sort()
        for i, token in enumerate(unique_tokens):
            self.token_to_id[token] = i
            self.id_to_token.append(token)
        return self

    def transform(self, x):
        if isinstance(x, str):
            x = [x]
        return [self.token_to_id[token] for token in np.array(x).flatten()]

    def __len__(self):
        return len(self.id_to_token)

    def __call__(self, x):
        return self.transform(x)

    def state_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Tokenizer":
        t = cls()
        t.token_to_id = state["token_to_id"]
        t.id_to_token = state["id_to_token"]
        return t


class DataFrameTokenizer:
    def __init__(self, df: pl.DataFrame, pad_length=8):
        self.rec_id_tokenizer = Tokenizer().fit(df["rec_id"].explode().unique())
        self.concentration_tokenizer = Tokenizer().fit(
            df["concentration"].explode().unique()
        )
        self.cell_type_tokenizer = Tokenizer().fit(df["cell_type"].unique())
        self.image_type_tokenizer = Tokenizer().fit(df["image_type"].unique())
        self.experiment_tokenizer = Tokenizer().fit(df["experiment_label"].unique())
        self.well_address_tokenizer = Tokenizer().fit(df["well_address"].unique())
        self.pad_length = pad_length

    def transform(self, row: dict[str, list[str]]):
        rec_id = self.rec_id_tokenizer(row["rec_id"])
        concentration = self.concentration_tokenizer(row["concentration"])

        return {
            "rec_id": F.pad(
                torch.tensor(rec_id, dtype=torch.long),
                (0, self.pad_length - len(rec_id)),
            ),
            "concentration": F.pad(
                torch.tensor(concentration, dtype=torch.long),
                (0, self.pad_length - len(concentration)),
            ),
            "comp_mask": F.pad(
                torch.ones(len(rec_id), dtype=torch.long),
                (0, self.pad_length - len(rec_id)),
            ).to(torch.bool),
            "cell_type": torch.tensor(
                self.cell_type_tokenizer(row["cell_type"])[0], dtype=torch.long
            ),
            "image_type": torch.tensor(
                self.image_type_tokenizer(row["image_type"])[0], dtype=torch.long
            ),
            "experiment_label": torch.tensor(
                self.experiment_tokenizer(row["experiment_label"])[0], dtype=torch.long
            ),
            "well_address": torch.tensor(
                self.well_address_tokenizer(row["well_address"])[0], dtype=torch.long
            ),
        }

    def __call__(self, row):
        return self.transform(row)

    def state_dict(self) -> dict:
        return {
            "rec_id": self.rec_id_tokenizer.state_dict(),
            "concentration": self.concentration_tokenizer.state_dict(),
            "cell_type": self.cell_type_tokenizer.state_dict(),
            "image_type": self.image_type_tokenizer.state_dict(),
            "experiment": self.experiment_tokenizer.state_dict(),
            "well_address": self.well_address_tokenizer.state_dict(),
            "pad_length": self.pad_length,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DataFrameTokenizer":
        t = object.__new__(cls)
        t.rec_id_tokenizer = Tokenizer.from_state_dict(state["rec_id"])
        t.concentration_tokenizer = Tokenizer.from_state_dict(state["concentration"])
        t.cell_type_tokenizer = Tokenizer.from_state_dict(state["cell_type"])
        t.image_type_tokenizer = Tokenizer.from_state_dict(state["image_type"])
        t.experiment_tokenizer = Tokenizer.from_state_dict(state["experiment"])
        t.well_address_tokenizer = Tokenizer.from_state_dict(state["well_address"])
        t.pad_length = state["pad_length"]
        return t
