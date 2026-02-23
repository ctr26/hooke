"""Diffusion transformer model architecture.
Forked from github.com/facebookresearch/DiT (MIT license)."""

import functools
from typing import Literal

import numpy as np
import ornamentalist
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed  # type: ignore
from hooke_forge.model.layers import Attention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            qk_norm=True,
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-5)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,  # type: ignore
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.context_norm = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-5)

    def forward(self, x, c):
        c = self.context_norm(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)  # type: ignore

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # type: ignore
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # type: ignore

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.weight, 0)  # type: ignore
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(N, T, patch_size**2 * C) -> (N, C, H, W)"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))  # type: ignore
        return imgs

    def forward(self, x, c):
        """Forward pass of DiT.
        x: (B, C, H, W) float32 tensor of spatial inputs (images or latent representations of images)
        c: (B, hidden_size) float32 tensor of conditioning inputs used as the residual stream."""
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x



class ConditionedMLP(nn.Module):
    """MLP vector field with concatenation-based conditioning.

    Three-stream architecture matching the hooke-predict BasicFMModule:
    1. mlp_xt: projects input data x into latent space
    2. mlp_c:  projects conditioning vector into latent space
    3. mlp_ut: maps concatenated [xt_emb, c_emb] to output velocity
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        xt_layers: int = 2,
        c_layers: int = 2,
        ut_layers: int = 2,
    ):
        super().__init__()
        self.mlp_xt = self._build_mlp(data_dim, hidden_dim, latent_dim, xt_layers)
        self.mlp_c = self._build_mlp(cond_dim, hidden_dim, latent_dim, c_layers)
        self.mlp_ut = self._build_mlp(2 * latent_dim, hidden_dim, data_dim, ut_layers)
        self._zero_init_output()

    @staticmethod
    def _build_mlp(in_dim, hidden_dim, out_dim, n_layers):
        """[Linear -> GELU] x n_layers -> Linear (no activation on final layer)."""
        layers = []
        dim_in = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim_in, hidden_dim), nn.GELU()]
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, out_dim))
        return nn.Sequential(*layers)

    def _zero_init_output(self):
        last = self.mlp_ut[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        xt_emb = self.mlp_xt(x)
        c_emb = self.mlp_c(cond)
        return self.mlp_ut(torch.cat([xt_emb, c_emb], dim=-1))

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


@ornamentalist.configure(name="model")
def get_model_cls(
    name: Literal[
        "DiT-XL/2", "DiT-XL/4", "DiT-XL/8",
        "DiT-L/2", "DiT-L/4", "DiT-L/8",
        "DiT-B/2", "DiT-B/4", "DiT-B/8",
        "DiT-S/2", "DiT-S/4", "DiT-S/8",
    ] = ornamentalist.Configurable["DiT-XL/2"],
) -> functools.partial[DiT]:  # fmt: skip
    model_variant, patch_size = name.split("/")
    match model_variant:
        case "DiT-S":
            depth = 12
            hidden_size = 384
            num_heads = 6
        case "DiT-B":
            depth = 12
            hidden_size = 768
            num_heads = 12
        case "DiT-L":
            depth = 24
            hidden_size = 1024
            num_heads = 16
        case "DiT-XL":
            depth = 28
            hidden_size = 1152
            num_heads = 16
        case _:
            raise ValueError(f"Unknown model name: {name}")

    patch_size = int(patch_size)
    assert patch_size in [2, 4, 8], f"Invalid patch size: {patch_size}"
    return functools.partial(
        DiT,
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
    )


#################################################################################
#                             ConditionedMLP Configs                            #
#################################################################################


@ornamentalist.configure(name="tx_model")
def get_tx_model_cls(
    name: Literal["TX-S", "TX-M", "TX-L"] = ornamentalist.Configurable["TX-S"],
) -> functools.partial[ConditionedMLP]:  # fmt: skip
    match name:
        case "TX-S":
            hidden_dim = 256
            latent_dim = 256
            xt_layers = 2
            c_layers = 2
            ut_layers = 2
        case "TX-M":
            hidden_dim = 512
            latent_dim = 512
            xt_layers = 3
            c_layers = 3
            ut_layers = 3
        case "TX-L":
            hidden_dim = 1024
            latent_dim = 512
            xt_layers = 4
            c_layers = 4
            ut_layers = 4
        case _:
            raise ValueError(f"Unknown TX model name: {name}")

    return functools.partial(
        ConditionedMLP,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        xt_layers=xt_layers,
        c_layers=c_layers,
        ut_layers=ut_layers,
    )
