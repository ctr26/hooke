"""Conditioning layers for the DiT generator."""

import math

from hooke_forge.model.tokenizer import MetaDataConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
from hooke_forge.model.layers import TransformerCore

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

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        rec_id_dim,
        concentration_dim,
        cell_type_dim,
        experiment_dim,
        assay_type_dim,
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
        self.assay_type_embedder = LabelEmbedder(
            num_classes=assay_type_dim, hidden_size=hidden_size, dropout_prob=0.0
        )
        self.well_address_embedder = LabelEmbedder(
            num_classes=well_address_dim,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
        )

        self.transformer = TransformerCore(
            n_embd=hidden_size, n_layer=12, n_head=16, dropout=0.0
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
        nn.init.normal_(self.assay_type_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.well_address_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.class_token, std=0.02)

    def forward(
        self,
        rec_id,
        concentration,
        cell_type,
        experiment_label,
        assay_type,
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
                self.assay_type_embedder(assay_type, train=self.training).unsqueeze(1),
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

def get_transformer_encoder(hidden_size: int, metadata_config: MetaDataConfig = MetaDataConfig()):
    return TransformerEncoder(
        hidden_size=hidden_size,
        rec_id_dim=metadata_config.rec_id_dim,
        concentration_dim=metadata_config.concentration_dim,
        cell_type_dim=metadata_config.cell_type_dim,
        experiment_dim=metadata_config.experiment_dim,
        assay_type_dim=metadata_config.assay_type_dim,
        well_address_dim=metadata_config.well_address_dim,
    )

