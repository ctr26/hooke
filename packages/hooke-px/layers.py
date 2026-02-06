"""
Modified from https://github.com/karpathy/nanoGPT/blob/master/model.py

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, bias, is_causal, dropout, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(n_embd // n_head)
        self.k_norm = nn.LayerNorm(n_embd // n_head)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.is_causal = is_causal

    def forward(self, x, mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = self.k_norm(
            k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = self.q_norm(
            q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Reshape mask from (B, T) to (B, 1, 1, T) for broadcasting with attention scores
        if mask is not None:
            mask = mask[:, None, None, :]
        y = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal,
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SetAttention(nn.Module):
    """Self-attention layer that prefers FlashAttention-2 if available."""

    def __init__(
        self,
        n_template=32,
        n_head=8,
        n_embd=512,
        bias=True,
        dropout=0.0,
        temp_q_init_scale=0.02,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_template = n_template
        self.head_dim = n_embd // n_head
        self.scale = 1 / math.sqrt(self.head_dim)

        self.template_q = nn.Parameter(
            torch.randn(n_head, n_template, n_embd // n_head) * temp_q_init_scale
        )

        self.kv = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.template_q.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_template, C)
        kv = self.kv(x)  # (B, T, 2·C)
        kv = kv.view(B, T, 2, self.n_head, self.head_dim).transpose(1, 3)
        k, v = kv.unbind(dim=2)  # each (B, nh, T, hd)

        y = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, self.n_template, C)  # (B, T, C)
        return self.resid_dropout(self.proj(y))


class SetBlock(nn.Module):
    """Transformer block with optional gradient checkpointing."""

    def __init__(
        self,
        n_template=32,
        n_head=8,
        n_embd=512,
        bias=True,
        dropout=0.0,
        temp_q_init_scale=0.02,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_template = n_template
        self.n_embd = n_embd
        self.bias = bias
        self.dropout = dropout
        self.temp_q_init_scale = temp_q_init_scale
        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.attn = SetAttention(
            n_template=n_template,
            n_head=n_head,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            temp_q_init_scale=temp_q_init_scale,
        )
        self.ln2 = LayerNorm(n_embd, bias=bias)
        self.ffwd = MLP(n_embd, bias, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(
            self.ln1(x)
        )  # no residual connection here because we're changing the dimensions
        x = x + self.ffwd(self.ln2(x))
        return x


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, bias, is_causal, dropout, n_head):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttention(n_embd, bias, is_causal, dropout, n_head)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 4096,
        n_layer: int = 3,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
        bias: bool = True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        is_causal: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, n_embd)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [Block(n_embd, bias, is_causal, dropout, n_head) for _ in range(n_layer)]
        )
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.drop(x)
        for block in self.layers:
            x = block(x, mask)
        return x


class TransformerCore(nn.Module):
    def __init__(
        self,
        *,
        n_layer: int = 3,
        n_head: int = 8,
        n_embd: int = 512,
        dropout: float = 0.0,
        bias: bool = True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        is_causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(n_embd, bias, is_causal, dropout, n_head) for _ in range(n_layer)]
        )
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        for block in self.layers:
            x = block(x, mask=mask)
        return x


class SetAutoencoder(nn.Module):
    def __init__(
        self,
        inp_dim,
        inp_length,
        downsample_factor=16,
        n_layer=3,
        n_head=8,
        n_embd=512,
        bias=True,
        dropout=0.0,
        temp_q_init_scale=0.02,
    ):
        super().__init__()
        self.input_proj = nn.Linear(inp_dim, n_embd)
        self.encoder = SetBlock(
            n_template=inp_length // downsample_factor,
            n_head=n_head,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            temp_q_init_scale=temp_q_init_scale,
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    n_head=n_head,
                    n_embd=n_embd,
                    bias=bias,
                    dropout=dropout,
                    is_causal=False,
                )
                for _ in range(n_layer)
            ]
        )
        self.decoder = SetBlock(
            n_template=inp_length,
            n_head=n_head,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            temp_q_init_scale=temp_q_init_scale,
        )
        self.output_proj = nn.Linear(n_embd, inp_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        for block in self.layers:
            x = block(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x


# Example usage
if __name__ == "__main__":
    import time

    model = SetAutoencoder(
        inp_dim=1,
        inp_length=4096,
        downsample_factor=16,
        n_layer=3,
        n_head=8,
        n_embd=512,
        bias=True,
        dropout=0.0,
        temp_q_init_scale=0.02,
    ).cuda()
    x = torch.randn(32, 4096, 1).cuda()
    start_time = time.time()
    for _ in range(10):
        y = model(x)
        del y
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 10} seconds per iteration")
