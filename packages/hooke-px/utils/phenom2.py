# type: ignore
"""
Standalone script to load Phenom-2 model without photosynthetic dependencies.
Only depends on torch and standard libraries.
"""

import math
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, VisionTransformer

DEFAULT_CHECKPOINT_PATH = "/mnt/ps/home/CORP/jason.hartford/project/Janus/artifacts/model-k2gyy7kr:v29/model.ckpt"
# ============================================================================
# Positional Encodings (from photosynthetic/training/nn/vit_positional_encodings.py)
# ============================================================================


def generate_2d_sincos_pos_embeddings(
    embedding_dim: int,
    length: int,
    scale: float = 10000.0,
    use_class_token: bool = True,
    num_modality: int = 1,
) -> torch.nn.Parameter:
    """Generate 2Dimensional sin/cosine positional embeddings"""
    linear_positions = torch.arange(length, dtype=torch.float32)
    height_mesh, width_mesh = torch.meshgrid(
        linear_positions, linear_positions, indexing="ij"
    )
    positional_dim = embedding_dim // 4  # accommodate h and w x cos and sin embeddings
    positional_weights = (
        torch.arange(positional_dim, dtype=torch.float32) / positional_dim
    )
    positional_weights = 1.0 / (scale**positional_weights)

    height_weights = torch.outer(height_mesh.flatten(), positional_weights)
    width_weights = torch.outer(width_mesh.flatten(), positional_weights)

    positional_encoding = torch.cat(
        [
            torch.sin(height_weights),
            torch.cos(height_weights),
            torch.sin(width_weights),
            torch.cos(width_weights),
        ],
        dim=1,
    )[None, :, :]

    # repeat positional encoding for multiple channel modalities
    positional_encoding = positional_encoding.repeat(1, num_modality, 1)

    if use_class_token:
        class_token = torch.zeros([1, 1, embedding_dim], dtype=torch.float32)
        positional_encoding = torch.cat([class_token, positional_encoding], dim=1)

    positional_encoding = torch.nn.Parameter(positional_encoding, requires_grad=False)
    return positional_encoding


# ============================================================================
# Normalizer (from photosynthetic/training/nn/modules/normalizer.py)
# ============================================================================


class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        return pixels / 255.0


# ============================================================================
# VIT Factory Functions (from photosynthetic/training/nn/modules/vit.py)
# ============================================================================


def vit_gigantic_patch16_256(**kwargs):
    """vit-G/16 configuration"""
    # Import ParallelScalingBlock from timm
    from timm.models.vision_transformer import ParallelScalingBlock

    default_kwargs = dict(
        img_size=256,
        in_chans=6,
        num_classes=0,
        fc_norm=None,
        class_token=True,
        patch_size=16,
        embed_dim=1664,  # vit-G
        mlp_ratio=64 / 13,  # vit-G
        depth=48,  # vit-G
        num_heads=16,
        drop_path_rate=0.6,
        init_values=0.0001,
        block_fn=ParallelScalingBlock,  # vit-22b speed optimization
        qkv_bias=False,  # vit-22b speed optimization
        qk_norm=True,  # vit-22b learning optimization
    )
    for k, v in kwargs.items():
        default_kwargs[k] = v
    return VisionTransformer(**default_kwargs)


def sincos_positional_encoding_vit(
    vit_backbone: VisionTransformer, scale: float = 10000.0
) -> VisionTransformer:
    """Attaches no-grad sin-cos positional embeddings to a pre-constructed ViT backbone model."""
    # length: number of tokens along height or width of image after patching (assuming square)
    length = (
        vit_backbone.patch_embed.img_size[0] // vit_backbone.patch_embed.patch_size[0]
    )
    pos_embeddings = generate_2d_sincos_pos_embeddings(
        vit_backbone.embed_dim,
        length=length,
        scale=scale,
        use_class_token=vit_backbone.cls_token is not None,
    )
    # note, if the model had weight_init == 'skip', this might get overwritten
    vit_backbone.pos_embed = pos_embeddings
    return vit_backbone


# ============================================================================
# MAE Components (from photosynthetic/training/models/mae.py)
# ============================================================================


def transformer_random_masking(
    x: torch.Tensor, mask_ratio: float, constant_noise: Union[torch.Tensor, None] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random mask patches per sample

    Parameters
    ----------
    x : token tensor (N, L, D)
    mask_ratio: float - ratio of image to mask
    constant_noise: None, if provided should be a tensor of shape (N, L) to produce consistent masks

    Returns
    -------
    x_masked : sub-sampled version of x ( int(mask_ratio * N), L, D)
    mask : binary mask indicated masked tokens (1 where masked) (N, L)
    ind_restore : locations of masked tokens, needed for decoder
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    # use random noise to generate batch based random masks
    if constant_noise is not None:
        noise = constant_noise
    else:
        noise = torch.rand(N, L, device=x.device)

    shuffled_tokens = torch.argsort(noise, dim=1)  # shuffled index
    ind_restore = torch.argsort(shuffled_tokens, dim=1)  # unshuffled index

    # get masked input
    tokens_to_keep = shuffled_tokens[:, :len_keep]  # keep the first len_keep indices
    x_masked = torch.gather(
        x, dim=1, index=tokens_to_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # get binary mask used for loss masking: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(
        mask, dim=1, index=ind_restore
    )  # unshuffle to get the binary mask

    return x_masked, mask, ind_restore


class MAEEncoder(nn.Module):
    def __init__(
        self,
        vit_backbone: VisionTransformer,
        max_in_chans: int = 6,
        bottleneck: Union[nn.Identity, nn.Module] = nn.Identity(),
        channel_agnostic: bool = False,
    ) -> None:
        super().__init__()
        # For simplicity, we ignore channel_agnostic functionality in this standalone version
        self.vit_backbone = vit_backbone
        self.max_in_chans = max_in_chans
        self.bottleneck = bottleneck
        self.channel_agnostic = channel_agnostic

        # changing fc_norm in backbone to match bottleneck, if one exists
        if not isinstance(self.bottleneck, nn.Identity):
            if isinstance(self.vit_backbone.fc_norm, nn.LayerNorm):
                self.vit_backbone.fc_norm = nn.LayerNorm(self.bottleneck.embed_dim)

    @property
    def embed_dim(self) -> int:
        if isinstance(self.bottleneck, nn.Identity):
            return int(self.vit_backbone.embed_dim)
        return self.bottleneck.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit_backbone.forward_features(x)
        x = self.bottleneck(x)
        x = self.vit_backbone.forward_head(x)
        return x

    def forward_masked(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        constant_noise: Union[torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.vit_backbone.patch_embed(x)
        x = self.vit_backbone._pos_embed(x)  # adds class token
        x_ = x[:, 1:, :]  # no class token
        x_, mask, ind_restore = transformer_random_masking(
            x_, mask_ratio, constant_noise
        )  # ONLY DIFFERENCE FROM VisionTransformer.forward_features
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add class token
        x = self.vit_backbone.norm_pre(x)
        x = self.vit_backbone.blocks(x)
        x = self.vit_backbone.norm(x)
        x = self.bottleneck(x)
        return x, mask, ind_restore


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embeddings = None  # to be overwritten by MAE class
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embeddings
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_masked(
        self, x: torch.Tensor, ind_restore: torch.Tensor
    ) -> torch.Tensor:
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ind_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # remove class token
        x_ = torch.gather(
            x_, dim=1, index=ind_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # add class token

        x = x + self.pos_embeddings  # type: ignore[operator]
        x = self.blocks(x)
        x = self.norm(x)
        return x  # type: ignore[no-any-return]


class MAEStandalone(nn.Module):
    """Standalone MAE model without training infrastructure"""

    def __init__(
        self,
        mask_ratio: float,
        encoder: MAEEncoder,
        decoder: MAEDecoder,
        input_norm: torch.nn.Module,
        norm_pix_loss: bool = False,
        crop_size: int = -1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mask_ratio = mask_ratio
        self.encoder = encoder
        self.in_chans = self.encoder.max_in_chans
        self.decoder = decoder
        self.input_norm = input_norm
        self.norm_pix_loss = norm_pix_loss

        self.crop_size = (
            encoder.vit_backbone.patch_embed.img_size[0]
            if crop_size == -1
            else crop_size
        )
        self.patch_size = int(self.encoder.vit_backbone.patch_embed.patch_size[0])

        # projection layer between the encoder and decoder
        self.encoder_decoder_proj = nn.Linear(
            self.encoder.embed_dim, self.decoder.embed_dim, bias=True
        )

        self.decoder_pred = nn.Linear(
            self.decoder.embed_dim,
            self.patch_size**2
            * (1 if self.encoder.channel_agnostic else self.in_chans),
            bias=True,
        )

        # overwrite decoder pos embeddings based on encoder params
        self.decoder.pos_embeddings = generate_2d_sincos_pos_embeddings(
            self.decoder.embed_dim,
            length=self.encoder.vit_backbone.patch_embed.grid_size[0],
            use_class_token=self.encoder.vit_backbone.cls_token is not None,
            num_modality=1,  # simplified for non-channel-agnostic case
        )

    @staticmethod
    def decode_to_reconstruction(
        encoder_latent: torch.Tensor,
        ind_restore: torch.Tensor,
        proj: torch.nn.Module,
        decoder: MAEDecoder,
        pred: torch.nn.Module,
    ) -> torch.Tensor:
        """Feed forward the encoder latent through the decoders necessary projections and transformations."""
        decoder_latent_projection = proj(
            encoder_latent
        )  # projection from encoder.embed_dim to decoder.embed_dim
        decoder_tokens = decoder.forward_masked(
            decoder_latent_projection, ind_restore
        )  # decoder.embed_dim output
        predicted_reconstruction = pred(
            decoder_tokens
        )  # linear projection to input dim
        return predicted_reconstruction[:, 1:, :]  # drop class token

    def forward(
        self, imgs: torch.Tensor, constant_noise: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = self.input_norm(imgs)
        latent, mask, ind_restore = self.encoder.forward_masked(
            imgs, self.mask_ratio, constant_noise
        )  # encoder blocks
        reconstruction = self.decode_to_reconstruction(
            latent,
            ind_restore,
            self.encoder_decoder_proj,
            self.decoder,
            self.decoder_pred,
        )
        return latent, reconstruction, mask

    def set_example_input_array(self, batch_size: int = 2) -> None:
        """Set example input for model summary"""
        self.example_input_array = torch.randint(
            low=0,
            high=255,
            size=(batch_size, self.in_chans, self.crop_size, self.crop_size),
            dtype=torch.uint8,
        )

    @staticmethod
    def flatten_images(
        img: torch.Tensor, patch_size: int = 8, channel_agnostic: bool = False
    ) -> torch.Tensor:
        """
        Flattens 2D images into tokens with the same pixel values

        Parameters
        ----------
        img : input image tensor (N, C, H, W)
        patch_size : size of the patch
        channel_agnostic : whether the model is channel agnostic

        Returns
        -------
        flattened_img: flattened image tensor (N, L, C * patch_size**2) such that the values
        within a patch first increase along the channel axis, then along the H axis, then along the W axis
        """
        if (img.shape[2] != img.shape[3]) or (img.shape[2] % patch_size != 0):
            raise ValueError(
                "image H must equal image W and be divisible by patch_size"
            )
        in_chans = img.shape[1]

        h = w = int(img.shape[2] // patch_size)
        x = img.reshape(shape=(img.shape[0], in_chans, h, patch_size, w, patch_size))

        if channel_agnostic:
            x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # NCHPWQ -> NCHWPQ
            x = x.reshape(shape=(img.shape[0], in_chans * h * w, int(patch_size**2)))
        else:
            x = torch.permute(x, (0, 2, 4, 3, 5, 1))  # NCHPWQ -> NHWPQC
            x = x.reshape(shape=(img.shape[0], h * w, int(patch_size**2 * in_chans)))
        return x

    @staticmethod
    def unflatten_tokens(
        tokens: torch.Tensor, patch_size: int = 8, num_modalities: int = 1
    ) -> torch.Tensor:
        """
        Unflattens tokens (N,L,C * patch_size**2) into image tensor (N,C,H,W) with the pixel values.
        The input tensor is assumed to be in the same format as the output of flatten_images.

        Parameters
        ----------
        tokens : input token tensor (N,L,C * patch_size**2)
        patch_size : size of the patch
        num_modalities : number of modalities
            Determines if the operations are channel agnostic or not. Default: 1, set to number of input channels for
            channel agnostic mode

        Returns
        -------
        img: image tensor (N,C,H,W)
        """

        h = w = int(math.sqrt(tokens.shape[1] // num_modalities))
        if h * w != (tokens.shape[1] // num_modalities):
            raise ValueError("sqrt of number of tokens not integer")

        if num_modalities > 1:
            x = tokens.reshape(
                shape=(tokens.shape[0], -1, h, w, patch_size, patch_size)
            )
            x = torch.permute(x, (0, 1, 2, 4, 3, 5))  # NCHWPQ -> NCHPWQ
        else:
            x = tokens.reshape(
                shape=(tokens.shape[0], h, w, patch_size, patch_size, -1)
            )
            x = torch.permute(x, (0, 5, 1, 3, 2, 4))  # NHWPQC -> NCHPWQ
        img = x.reshape(shape=(x.shape[0], -1, h * patch_size, h * patch_size))

        return img


# ============================================================================
# Model Loading Functions
# ============================================================================


def build_phenom2_model() -> MAEStandalone:
    """Build Phenom-2 model with hardcoded configuration (matching models/phenom-2.yaml)"""

    # Build ViT backbone
    vit_backbone = vit_gigantic_patch16_256(
        global_pool="avg", patch_size=8, dynamic_img_size=True
    )

    # Add positional encodings
    vit_backbone = sincos_positional_encoding_vit(vit_backbone)

    # Build encoder
    encoder = MAEEncoder(vit_backbone=vit_backbone, channel_agnostic=False)

    # Build decoder
    decoder = MAEDecoder(
        embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    # Build input normalization
    input_norm = nn.Sequential(
        Normalizer(),
        nn.InstanceNorm2d(num_features=6, affine=False, track_running_stats=False),
    )

    # Build complete model
    model = MAEStandalone(
        mask_ratio=0.75,
        encoder=encoder,
        decoder=decoder,
        input_norm=input_norm,
        norm_pix_loss=False,
    )

    return model


def load_phenom2(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH, device: torch.device | None = None
) -> MAEStandalone:
    """
    Load Phenom-2 model from checkpoint file.

    Args:
        checkpoint_path: Path to the model checkpoint (.pickle or .ckpt file)
        device: Device to load model on ('cpu', 'cuda', etc.). Auto-detected if None.

    Returns:
        Loaded MAEStandalone model in eval mode
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = build_phenom2_model()

    # Load checkpoint
    loaded_checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=device
    )

    # Extract state dict (handle different checkpoint formats)
    if "state_dict" in loaded_checkpoint:
        state_dict = loaded_checkpoint["state_dict"]
    else:
        state_dict = loaded_checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)

    # Set to eval mode
    model.eval()

    # Move to device
    model.to(device)

    # Set example input for model summary
    model.set_example_input_array(batch_size=2)

    return model


# ============================================================================
# Example Reconstruction Plots to verify everything is working
# ============================================================================
def rescale_intensity(
    arr: torch.Tensor, bounds=(0.5, 99.5), out_range=(0.0, 1.0)
) -> torch.Tensor:
    arr = arr.float() / 255
    sample = arr.flatten()[::100]
    percentiles = torch.quantile(
        sample, torch.tensor([bounds[0] / 100.0, bounds[1] / 100.0])
    )
    arr = torch.clamp(arr, percentiles[0], percentiles[1])
    arr = (arr - percentiles[0]) / (percentiles[1] - percentiles[0])
    arr = arr * (out_range[1] - out_range[0]) + out_range[0]
    return arr


def to_rgb(img: torch.Tensor, dtype=torch.float32) -> torch.Tensor:  # type: ignore[no-untyped-def]
    num_channels_required = 6
    b, num_channels, length, width = img.shape  # b x c x l x w
    # Backfill the image arbitrarily with zeros. this could be more faithful to the image if this function
    # knows exactly the channels-to-keep, but the visualization difference would be minor
    # while we would need to propagate that parameter through several classes functions to get here.
    # So instead we approximate by just filling in the image with zeros for missing channels.
    prepped_img = torch.zeros(
        b, num_channels_required, length, width, dtype=img.dtype, device=img.device
    )
    if num_channels < num_channels_required:
        prepped_img[:, :num_channels, :, :] += img
    elif num_channels > num_channels_required:
        prepped_img += img[:, :num_channels_required, :, :]
    else:
        prepped_img += img
    # color mapping
    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    yellow = [1, 1, 0]
    magenta = [1, 0, 1]
    cyan = [0, 1, 1]
    rgb_map = torch.tensor(
        [blue, green, red, cyan, magenta, yellow],
        dtype=dtype,
        device=prepped_img.device,
    )
    rgb_img: torch.FloatTensor = (
        torch.einsum(  # type: ignore[assignment]
            "nchw,ct->nthw",
            prepped_img.to(dtype=dtype),
            rgb_map,
        )
        / 3.0
    )
    # return rgb_img
    return rescale_intensity(rgb_img, bounds=(0.1, 99.9))


def apply_mask_to_image(img, mask, patch_size=8):
    N, C, H, W = img.shape
    assert H == W, "The height and width of the image should be equal"
    L = (H // patch_size) ** 2
    assert mask.shape == (N, L), "The mask should have shape (N, L)"

    h = w = H // patch_size
    img_patches = img.reshape(N, C, h, patch_size, w, patch_size)
    img_patches = img_patches.permute(0, 2, 4, 3, 5, 1).reshape(
        N, h * w, patch_size, patch_size, C
    )
    masked_img_patches = img_patches.clone()
    masked_img_patches[mask.bool()] = 0
    masked_img_patches = masked_img_patches.reshape(
        N, h, w, patch_size, patch_size, C
    ).permute(0, 5, 1, 3, 2, 4)
    masked_img = masked_img_patches.reshape(N, C, H, W)
    return masked_img


def plot_reconstruction(model_ph2):
    import zarr
    from matplotlib import pyplot as plt

    # Load example images so we don't need a data loader
    paths = [
        "/rxrx/data/microscopy/zarr_512_DSx2_8bit_zstd_3_2/pheno-WholeGenome014-H-a/Plate1/Order4013/Read157/AA02_s1.zarr",
        "/rxrx/data/microscopy/zarr_512_DSx2_8bit_zstd_3_2/pheno-WholeGenome014-H-a/Plate1/Order4013/Read157/S11_s1.zarr",
        "/rxrx/data/microscopy/zarr_512_DSx2_8bit_zstd_3_2/pheno-WholeGenome014-H-a/Plate8/Order4013/Read160/H29_s1.zarr",
    ]
    top, left = 896, 896
    convert_to_tensor = lambda x: torch.from_numpy(x).permute(2, 0, 1).contiguous()  # noqa
    img = [
        convert_to_tensor(
            zarr.open_array(path, mode="r")[top : top + 256, left : left + 256]
        )
        for path in paths
    ]
    img = torch.stack(img).to(next(model_ph2.parameters()).device)

    with torch.no_grad():
        latent, reconstruction, mask = model_ph2(img.clone())
        reconstruction = model_ph2.unflatten_tokens(reconstruction, patch_size=8)

    selfstandardizer = torch.nn.modules.InstanceNorm2d(
        num_features=6, affine=False, track_running_stats=False
    )
    original = to_rgb(selfstandardizer(img.cpu() / 255))
    predicted = to_rgb(reconstruction.cpu())
    masked_image = apply_mask_to_image(original.cpu(), mask.cpu())

    for i in range(3):
        plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

        # Display the original image on the left
        plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
        plt.imshow(original[i].transpose(2, 0))
        plt.title("Original")

        # Display the predicted image on the right
        plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(masked_image[i].transpose(2, 0))
        plt.title("Masked original")

        # Display the predicted image on the right
        plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
        plt.imshow(predicted[i].transpose(2, 0))
        plt.title("Predicted")

        plt.savefig(f"reconstruction_standalone_{i}.png")
        plt.close()


# ============================================================================
# Main Script
# ============================================================================


def main():
    """Main function to demonstrate model loading"""
    # Default checkpoint path from models.yaml (using the same path as original script)
    save_test_plots = True

    print("Loading Phenom-2 model...")
    try:
        model = load_phenom2(device="cuda" if torch.cuda.is_available() else "cpu")
        print("Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the checkpoint path is correct and accessible.")

    if save_test_plots:
        plot_reconstruction(model)

        # get embeddings:
        # with torch.no_grad():
        #    img = ...
        #    embeddings, _, _ = model.encoder.forward_masked(img, mask_ratio=0.0)
        #    print(embeddings.shape)


if __name__ == "__main__":
    main()
