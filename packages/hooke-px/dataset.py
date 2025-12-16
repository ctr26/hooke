import dataclasses
from typing import Callable
import logging
import numcodecs

import diffusers
import ornamentalist
import polars as pl
import torch
import zarr
import zarr.core
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2
from adaptor import DataFrameTokenizer

numcodecs.blosc.use_threads = False

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# encode these cols as ints in the range [0, num_classes)
# no harm in this being larger than the actual number of unique labels
NUM_PERTURBATIONS = 640_000 - 1  # -1 is because we will use an extra uncond token
NUM_EXPERIMENTS = 4160 - 1
NUM_CELL_TYPES = 36 - 1

IMG_SIZE = 256  # the size of image that the model expects
MAX_CROP_SIZE = 512  # max crop for multiscale training (will be resized to IMG_SIZE)
MIN_CROP_SIZE = 64  # min crop for multiscale training (will be resized to IMG_SIZE)


class CellPaintConverter:
    def __init__(self, device: torch.device):
        self._RGB_MAP = {
            1: {"rgb": [19.0, 0.0, 249.0], "range": [0.0, 51.0]},
            2: {"rgb": [42.0, 255.0, 31.0], "range": [0.0, 107.0]},
            3: {"rgb": [255.0, 0.0, 25.0], "range": [0.0, 64.0]},
            4: {"rgb": [45.0, 255.0, 252.0], "range": [0.0, 191.0]},
            5: {"rgb": [250.0, 0.0, 253.0], "range": [0.0, 89.0]},
            6: {"rgb": [254.0, 255.0, 40.0], "range": [0.0, 191.0]},
        }
        self._RANGES = (
            torch.tensor(
                [v["range"] for v in self._RGB_MAP.values()],
                device=device,
                dtype=torch.float32,
            )
            / 255.0
        )
        self._RGB_COLORS = (
            torch.tensor(
                [v["rgb"] for v in self._RGB_MAP.values()],
                device=device,
                dtype=torch.float32,
            )
            / 255.0
        )
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of 6-channel cell-paint images to RGB space.
        input: torch.uint8 (B, 6, H, W), range [0, 255]
        output: torch.uint8 (B, 3, H, W), range [0, 255]
        """
        assert x.ndim == 4, "Input must be a 4D batch tensor"
        assert x.shape[1] == 6, "Input must have 6 channels"
        assert x.dtype == torch.uint8, "Input must be uint8"
        device = x.device
        val_min = self._RANGES[:, 0].view(1, 6, 1, 1).to(device)
        val_max = self._RANGES[:, 1].view(1, 6, 1, 1).to(device)
        val_diff = (val_max - val_min).clamp(min=1e-6)

        rgb_colors = self._RGB_COLORS.to(device)
        x = x.to(torch.float32) / 255.0
        scaled_channels = (x - val_min) / val_diff
        x = torch.einsum("bchw,cr->brhw", scaled_channels, rgb_colors)
        x = (x * 255.0).clamp(0, 255).to(torch.uint8)
        return x


class StabilityCPEncoder:
    def __init__(self, device: torch.device, compile: bool = True):
        self.vae = diffusers.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")  # type: ignore
        self.vae.eval().requires_grad_(False).to(device)  # type: ignore
        self.scale = 0.18215  # magic constant from the DiT codebase

        if compile:
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of cell-paint images into the stable diffusion latent space.
        input: torch.uint8 [0, 255], (B, 6, H, W)
        output: torch.float32, (B, 8, H // 8, W // 8)"""
        x = x.to(torch.float32) / 127.5 - 1
        x0, x1 = torch.chunk(x, 2, dim=1)
        x0 = self.vae.encode(x0).latent_dist.sample()  # type: ignore
        x1 = self.vae.encode(x1).latent_dist.sample()  # type: ignore
        return torch.cat([x0, x1], dim=1) * self.scale

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of stable diffusion latents into cell-paint space.
        input: torch.float32, (B, 8, H // 8, W // 8)
        output: torch.uint8 [0, 255], (B, 6, H, W)"""
        x = x / self.scale
        x0, x1 = torch.chunk(x, 2, dim=1)
        x0 = self.vae.decode(x0).sample  # type: ignore
        x1 = self.vae.decode(x1).sample  # type: ignore
        x = torch.cat([x0, x1], dim=1)
        x = (x + 1) * 127.5
        x = x.clip(0, 255).to(torch.uint8)
        return x


def crop_zarr(
    zarr_array: zarr.core.Array, top: int, left: int, height: int, width: int
) -> torch.Tensor:
    """Takes a zarr array and returns a cropped uint8 torch tensor (C x H x W)
    (only loads the relevant crop into memory)."""
    zarr_array_cropped = zarr_array[top : (top + height), left : (left + width), :]
    tensor = torch.from_numpy(zarr_array_cropped)
    tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous()


def center_crop_zarr(zarr_array: zarr.core.Array, size: int) -> torch.Tensor:
    h, w, _ = zarr_array.shape
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return crop_zarr(zarr_array, top, left, size, size)


def random_crop_zarr(
    zarr_array: zarr.core.Array, size: int, border: int = 0
) -> torch.Tensor:
    h, w, _ = zarr_array.shape
    top = int(torch.randint(border, h - (size + border), (1,)).item())
    left = int(torch.randint(border, w - (size + border), (1,)).item())
    return crop_zarr(zarr_array, top, left, size, size)


def get_transforms(train: bool, size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if train:
        return v2.Compose(
            [
                v2.Resize(size=(size, size), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ]
        )
    else:
        return v2.Compose([v2.Resize(size=(size, size), antialias=True)])


class CellDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        tokenizer: DataFrameTokenizer,
        train: bool,
        size: int,
        multiscale: bool = False,
    ):
        required_cols = [
            "image_path",
            "cell_type",
            "experiment_label",
            "image_type",
            "well_address",
            "rec_id",
            "concentration",
        ]
        assert all(col in metadata.columns for col in required_cols), (
            f"metadata must have the following columns: {required_cols}"
        )
        self.metadata = metadata
        self.transforms = get_transforms(train, size)
        self.multiscale = multiscale
        self.size = size
        self.train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.metadata)

    def _get_fallback_sample(self):
        """Return a valid fallback sample with zeros when data loading fails."""
        tensor = torch.zeros(6, self.size, self.size, dtype=torch.uint8)
        # Use first row as fallback metadata (will be masked out anyway)
        fallback_row = self.metadata.row(0, named=True)
        return {
            "img": tensor,
            "meta": self.tokenizer(fallback_row),
        }

    def __getitem__(self, index: int):
        try:
            row = self.metadata.row(index, named=True)
            path = row["image_path"]
            # uint8 array (lazy loaded memory-mapped file)
            array = zarr.open_array(path, mode="r")
            with torch.inference_mode():
                if self.multiscale:
                    crop_size = 2 * int(
                        torch.randint(
                            MIN_CROP_SIZE // 2, MAX_CROP_SIZE // 2, (1,)
                        ).item()
                    )
                else:
                    crop_size = self.size

                if self.train:
                    tensor = random_crop_zarr(array, crop_size, border=256)
                else:
                    tensor = center_crop_zarr(array, crop_size)

                # C x H x W at this point. Will be stacked into
                # B x C x H x W by collate_fn in the dataloader
                tensor = self.transforms(tensor)
                if tensor.shape[0] == 3:
                    tensor = torch.cat([tensor, tensor], dim=0)

            sample = {
                "img": tensor,
                "meta": self.tokenizer(row),
            }
            if "zarr_index" in row:
                sample["zarr_index"] = row["zarr_index"]
            return sample
        except Exception as e:
            _log.warning(
                "Failed to load sample at index %d. Falling back to zeros. Error: %s",
                index,
                e,
            )
            return self._get_fallback_sample()


class HFCellDataset(torch.utils.data.Dataset):
    """CellDataset backed by HuggingFace Dataset for O(1) memory-mapped access.

    This dataset reads from a pre-processed HuggingFace Dataset cache that contains
    pre-tokenized metadata. This avoids the O(N) row access overhead of Polars
    DataFrames and enables efficient memory sharing across workers via mmap.

    Use scripts/prepare_hf_cache.py to create the cache from a parquet file.
    """

    def __init__(
        self,
        cache_dir: str,
        train: bool,
        size: int,
        pad_length: int = 8,
        multiscale: bool = False,
    ):
        from datasets import load_from_disk

        self.hf_dataset = load_from_disk(cache_dir)
        self.transforms = get_transforms(train, size)
        self.multiscale = multiscale
        self.size = size
        self.train = train
        self.pad_length = pad_length

    def __len__(self):
        return len(self.hf_dataset)

    def _get_fallback_sample(self):
        """Return a valid fallback sample with zeros when data loading fails."""
        tensor = torch.zeros(6, self.size, self.size, dtype=torch.uint8)
        row = self.hf_dataset[0]
        return {
            "img": tensor,
            "meta": {
                "rec_id": torch.tensor(row["rec_id"], dtype=torch.long),
                "concentration": torch.tensor(row["concentration"], dtype=torch.long),
                "comp_mask": torch.arange(self.pad_length) < row["rec_id_len"],
                "cell_type": torch.tensor(row["cell_type"], dtype=torch.long),
                "image_type": torch.tensor(row["image_type"], dtype=torch.long),
                "experiment_label": torch.tensor(
                    row["experiment_label"], dtype=torch.long
                ),
                "well_address": torch.tensor(row["well_address"], dtype=torch.long),
            },
        }

    def __getitem__(self, index: int):
        try:
            row = self.hf_dataset[index]  # O(1) memory-mapped access
            path = row["image_path"]

            # uint8 array (lazy loaded memory-mapped file)
            array = zarr.open_array(path, mode="r")
            with torch.inference_mode():
                if self.multiscale:
                    crop_size = 2 * int(
                        torch.randint(
                            MIN_CROP_SIZE // 2, MAX_CROP_SIZE // 2, (1,)
                        ).item()
                    )
                else:
                    crop_size = self.size

                if self.train:
                    tensor = random_crop_zarr(array, crop_size, border=256)
                else:
                    tensor = center_crop_zarr(array, crop_size)

                # C x H x W at this point. Will be stacked into
                # B x C x H x W by collate_fn in the dataloader
                tensor = self.transforms(tensor)
                if tensor.shape[0] == 3:
                    tensor = torch.cat([tensor, tensor], dim=0)

            # Build metadata dict (already tokenized and padded in cache)
            meta = {
                "rec_id": torch.tensor(row["rec_id"], dtype=torch.long),
                "concentration": torch.tensor(row["concentration"], dtype=torch.long),
                "comp_mask": torch.arange(self.pad_length) < row["rec_id_len"],
                "cell_type": torch.tensor(row["cell_type"], dtype=torch.long),
                "image_type": torch.tensor(row["image_type"], dtype=torch.long),
                "experiment_label": torch.tensor(
                    row["experiment_label"], dtype=torch.long
                ),
                "well_address": torch.tensor(row["well_address"], dtype=torch.long),
            }
            return {"img": tensor, "meta": meta}
        except Exception as e:
            _log.warning(
                "Failed to load sample at index %d. Falling back to zeros. Error: %s",
                index,
                e,
            )
            return self._get_fallback_sample()


@dataclasses.dataclass(frozen=True)
class MetaVocab:
    rec_id_dim: int
    concentration_dim: int
    cell_type_dim: int
    experiment_dim: int
    image_type_dim: int
    well_address_dim: int
    pad_length: int


@ornamentalist.configure()
def get_dataloaders(
    *,
    path: str = ornamentalist.Configurable[
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/pretraining_v2.parquet"
    ],
    batch_size: int = ornamentalist.Configurable[64],
    num_workers: int = ornamentalist.Configurable[16],
    pin_memory: bool = ornamentalist.Configurable[True],
    pad_length: int = 8,
    use_cache: bool = ornamentalist.Configurable[True],
) -> tuple[DataLoader, DataLoader, MetaVocab, DataFrameTokenizer]:
    import json
    from pathlib import Path

    cache_dir = Path(path).with_suffix(".cache")

    # Use HF cache if available and enabled
    if use_cache and cache_dir.exists() and (cache_dir / "tokenizer.json").exists():
        _log.info(f"Using HuggingFace Dataset cache at {cache_dir}")

        # Load tokenizer from cache
        with open(cache_dir / "tokenizer.json") as f:
            tokenizer = DataFrameTokenizer.from_state_dict(json.load(f))

        vocab = MetaVocab(
            rec_id_dim=len(tokenizer.rec_id_tokenizer),
            concentration_dim=len(tokenizer.concentration_tokenizer),
            cell_type_dim=len(tokenizer.cell_type_tokenizer),
            experiment_dim=len(tokenizer.experiment_tokenizer),
            image_type_dim=len(tokenizer.image_type_tokenizer),
            well_address_dim=len(tokenizer.well_address_tokenizer),
            pad_length=tokenizer.pad_length,
        )

        train_ds = HFCellDataset(
            str(cache_dir / "train"),
            train=True,
            size=IMG_SIZE,
            pad_length=pad_length,
        )
        val_ds = HFCellDataset(
            str(cache_dir / "val"),
            train=False,
            size=IMG_SIZE,
            pad_length=pad_length,
        )
    else:
        if use_cache:
            _log.info(f"No cache found at {cache_dir}, falling back to DataFrame")

        df = pl.read_parquet(path)
        train_df = df.filter(pl.col("split") == "train")
        val_df = df.filter(pl.col("split") == "valid")

        tokenizer = DataFrameTokenizer(df, pad_length=pad_length)
        vocab = MetaVocab(
            rec_id_dim=len(tokenizer.rec_id_tokenizer),
            concentration_dim=len(tokenizer.concentration_tokenizer),
            cell_type_dim=len(tokenizer.cell_type_tokenizer),
            experiment_dim=len(tokenizer.experiment_tokenizer),
            image_type_dim=len(tokenizer.image_type_tokenizer),
            well_address_dim=len(tokenizer.well_address_tokenizer),
            pad_length=tokenizer.pad_length,
        )

        train_ds = CellDataset(
            train_df,
            tokenizer=tokenizer,
            train=True,
            size=IMG_SIZE,
        )
        val_ds = CellDataset(
            val_df,
            tokenizer=tokenizer,
            train=False,
            size=IMG_SIZE,
        )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=8,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, vocab, tokenizer
