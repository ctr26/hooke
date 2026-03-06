import dataclasses
import logging
import random
from collections.abc import Callable
from pathlib import Path

import numcodecs
import numpy as np
import ornamentalist
import polars as pl
import torch
import zarr
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2

from hooke_forge.model.tokenizer import DataFrameTokenizer

numcodecs.blosc.use_threads = False

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# encode these cols as ints in the range [0, num_classes)
# no harm in this being larger than the actual number of unique labels
NUM_PERTURBATIONS = 640_000 - 1  # -1 is because we will use an extra uncond token
NUM_EXPERIMENTS = 4160 - 1
NUM_CELL_TYPES = 36 - 1

IMG_SIZE = 256  # the size of image that the model expects


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


def crop_zarr(zarr_array: zarr.Array, top: int, left: int, height: int, width: int) -> torch.Tensor:
    """Takes a zarr array and returns a cropped uint8 torch tensor (C x H x W)
    (only loads the relevant crop into memory)."""
    zarr_array_cropped = zarr_array[top : (top + height), left : (left + width), :]
    tensor = torch.from_numpy(zarr_array_cropped)
    tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous()


def center_crop_zarr(zarr_array: zarr.Array, size: int) -> torch.Tensor:
    h, w, _ = zarr_array.shape
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return crop_zarr(zarr_array, top, left, size, size)


def random_crop_zarr(zarr_array: zarr.Array, size: int, border: int = 0) -> torch.Tensor:
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


_TX_TARGET_SUM: float = 4_000.0  # library-size normalization target (matches hooke-predict)


def _normalize_log1p(x: torch.Tensor, target_sum: float = _TX_TARGET_SUM) -> torch.Tensor:
    """Library-size normalize then log1p-transform a 1-D expression vector.

    Replicates hooke-predict's ``TaskDatasetBase.transform()``:
      1. Scale counts so the row sums to ``target_sum`` (default 4,000).
      2. Apply log1p  →  output values in [0, ~8.3] for target_sum=4000.

    Without this step the raw zarr values (up to ~74k, row sum ~1M) produce
    flow-matching velocities with MSE ~60,000 instead of the expected O(1-10).
    """
    row_sum = x.sum()
    if row_sum == 0:
        return x
    x = (x / row_sum) * target_sum
    return torch.log1p(x)


class TxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pl.DataFrame,
        tokenizer: DataFrameTokenizer,
        zarr_path: str | Path = Path(  # noqa: E501
            "/rxrx/data/user/ali.denton/tmp/training_trek__v1_0/training_trek__v1_0_features.zarr"
        ),
        gene_subset_path: str | Path | None = None,
        train: bool = True,
    ):
        self.train = train
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.zarr_file = zarr.open(zarr_path)

        # Gene subset initialization
        self.gene_subset_path = gene_subset_path
        self.gene_mask = None
        self.hvg_indices = None

        if gene_subset_path is not None:
            self._load_gene_subset(gene_subset_path)

        if train:
            required_cols = [
                "zarr_row_idx",
                "batch_center",
                "is_negative_control",
                "cell_type",
                "experiment_label",
                "assay_type",
                "well_address",
                "rec_id",
                "concentration",
            ]
            assert all(col in metadata.columns for col in required_cols), (
                f"metadata must have the following columns: {required_cols}"
            )
            lookup_df = metadata.group_by("batch_center").agg(
                [pl.col("zarr_row_idx").filter(pl.col("is_negative_control")).alias("control_indices")]
            )
            control_map = dict(zip(lookup_df["batch_center"], lookup_df["control_indices"]))
            self.control_map = {k: v.to_list() for k, v in control_map.items()}
            self.skip = [k for k, v in self.control_map.items() if len(v) == 0]
            _log.warning(f"Skipping {len(self.skip)} batch centers with no controls")
            self.all_controls = self.metadata.filter(pl.col("is_negative_control"))["zarr_row_idx"].unique()

    def _load_gene_subset(self, gene_subset_path: str | Path) -> None:
        """Load pre-computed gene subset from .npz file."""
        try:
            data = np.load(gene_subset_path, allow_pickle=True)
            self.gene_mask = data["gene_mask"]
            self.hvg_indices = data["hvg_indices"]

            # Optional: log gene symbols for debugging
            if "gene_symbols" in data:
                gene_symbols = data["gene_symbols"]
                _log.info(f"Loaded gene subset with {len(self.hvg_indices)} features")
                _log.debug(f"First 10 genes: {gene_symbols[:10].tolist()}")
            else:
                _log.info(f"Loaded gene subset with {len(self.hvg_indices)} features")

            # Optional: log metadata for transparency
            if "metadata" in data:
                metadata = data["metadata"].item() if data["metadata"].ndim > 0 else data["metadata"]
                if isinstance(metadata, dict) and "config" in metadata:
                    config = metadata["config"]
                    _log.info(
                        f"Gene subset config: strategy={config.get('select_strategy', 'unknown')}, "
                        f"n_features={config.get('n_features', 'unknown')}"
                    )

        except Exception as e:
            _log.error(f"Failed to load gene subset from {gene_subset_path}: {e}")
            raise ValueError(f"Could not load gene subset file: {e}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.row(index, named=True)

        if not self.train:
            # Inference mode: only need metadata for conditioning
            sample = {"meta": self.tokenizer(row)}
            if "zarr_index" in row:
                sample["zarr_index"] = row["zarr_index"]
            return sample

        # Load features
        tx = torch.from_numpy(self.zarr_file[row["zarr_row_idx"]])

        # Apply gene subset filtering if configured
        if self.gene_subset_path is not None:
            tx = tx[self.gene_mask][self.hvg_indices]

        # Training mode: pair with control
        if row["batch_center"] in self.skip:
            control = self.metadata.row(random.choice(self.all_controls), named=True)
        else:
            control = self.metadata.row(random.choice(self.control_map[row["batch_center"]]), named=True)

        tx_control = torch.from_numpy(self.zarr_file[control["zarr_row_idx"]])
        if self.gene_subset_path is not None:
            tx_control = tx_control[self.gene_mask][self.hvg_indices]

        tx_raw = tx.clone()
        tx = _normalize_log1p(tx)
        tx_control = _normalize_log1p(tx_control)

        return {
            "tx": tx,
            "tx_raw": tx_raw,
            "tx_control": tx_control,
            "meta": self.tokenizer(row),
        }


@dataclasses.dataclass(frozen=True)
class MetaVocab:
    rec_id_dim: int
    concentration_dim: int
    cell_type_dim: int
    experiment_dim: int
    assay_type_dim: int
    well_address_dim: int
    pad_length: int


@ornamentalist.configure()
def get_dataloaders(
    *,
    path: str = ornamentalist.Configurable[
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/pretraining_v6.parquet"
    ],
    batch_size: int = ornamentalist.Configurable[64],
    num_workers: int = ornamentalist.Configurable[16],
    pin_memory: bool = ornamentalist.Configurable[True],
    pad_length: int = 8,
    tokenizer: DataFrameTokenizer | None = None,
) -> tuple[DataLoader, DataLoader, MetaVocab, DataFrameTokenizer]:
    """Get train and validation dataloaders.

    Args:
        tokenizer: Optional pre-existing tokenizer (e.g., loaded from checkpoint).
                   If provided, this tokenizer will be used instead of fitting a new one.
                   This is useful for finetuning on a subset of data while preserving
                   the original vocabulary.
    """

    df = pl.read_parquet(path)

    # Older parquets use `image_type` instead of `assay_type` — normalise to `assay_type`
    if "assay_type" not in df.columns and "image_type" in df.columns:
        _log.info("'assay_type' column not found; using 'image_type' as 'assay_type'")
        df = df.with_columns(pl.col("image_type").alias("assay_type"))
    elif "assay_type" not in df.columns:
        raise KeyError("assay_type missing from columns")

    train_df = df.filter(pl.col("split") == "train")
    val_df = df.filter(pl.col("split") == "valid_cp_iid")

    # Use provided tokenizer, or fit a new one on the data
    if tokenizer is None:
        tokenizer = DataFrameTokenizer(df, pad_length=pad_length)
    else:
        _log.info("Using provided tokenizer instead of fitting new one")
    vocab = MetaVocab(
        rec_id_dim=len(tokenizer.rec_id_tokenizer),
        concentration_dim=len(tokenizer.concentration_tokenizer),
        cell_type_dim=len(tokenizer.cell_type_tokenizer),
        experiment_dim=len(tokenizer.experiment_tokenizer),
        assay_type_dim=len(tokenizer.assay_type_tokenizer),
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
        prefetch_factor=12,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
        drop_last=False,
    )
    assert tokenizer is not None  # either provided or created above
    return train_loader, val_loader, vocab, tokenizer


@ornamentalist.configure(name="tx_data")
def get_tx_dataloaders(
    *,
    path: str = ornamentalist.Configurable[
        "/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/training_trek_v1_0_obs.parquet"
    ],
    zarr_path: str = ornamentalist.Configurable[
        "/rxrx/data/user/ali.denton/tmp/training_trek__v1_0/training_trek__v1_0_features.zarr"
    ],
    batch_size: int = ornamentalist.Configurable[256],
    num_workers: int = ornamentalist.Configurable[8],
    pin_memory: bool = ornamentalist.Configurable[True],
    val_split: str = ornamentalist.Configurable["valid_tx"],
    pad_length: int = 8,
    tokenizer: DataFrameTokenizer | None = None,
    gene_subset_path: str = ornamentalist.Configurable[""],
) -> tuple[DataLoader, DataLoader | None, DataFrameTokenizer]:
    """Get Tx train (and optionally validation) dataloaders.

    Args:
        tokenizer: Optional pre-existing tokenizer. If provided, it will be
                   used as-is (useful for joint training or finetuning).
    """
    df = pl.read_parquet(path)

    # Ensure the expected columns exist; add zarr_row_idx if missing
    if "zarr_row_idx" not in df.columns:
        df = df.with_row_index("zarr_row_idx")

    # Prepare rec_id / concentration from perturbations column if needed
    if "rec_id" not in df.columns and "perturbations" in df.columns:
        df = df.with_columns(
            rec_id=pl.col("perturbations").list.eval(pl.element().struct.field("source_id")),
            concentration=pl.col("perturbations").list.eval(
                pl.element().struct.field("concentration").cast(pl.Float64).cast(pl.String)
            ),
        )

    if "assay_type" not in df.columns:
        df = df.with_columns(pl.lit("trek").alias("assay_type"))

    if "split" not in df.columns:
        df = df.with_columns(pl.lit("train").alias("split"))

    train_df = df.filter(pl.col("split") == "train")

    # Use provided tokenizer, or fit a new one on the data
    if tokenizer is None:
        tokenizer = DataFrameTokenizer(df, pad_length=pad_length)
    else:
        _log.info("Using provided tokenizer for Tx data")

    train_ds = TxDataset(
        train_df,
        tokenizer=tokenizer,
        zarr_path=zarr_path,
        gene_subset_path=gene_subset_path if gene_subset_path else None,
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Validation loader (optional -- may not exist yet for all Tx datasets)
    val_loader = None
    val_df = df.filter(pl.col("split") == val_split)
    if len(val_df) > 0:
        val_ds = TxDataset(
            val_df,
            tokenizer=tokenizer,
            zarr_path=zarr_path,
            gene_subset_path=gene_subset_path if gene_subset_path else None,
        )
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=val_sampler,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        _log.warning("No Tx validation data found for split='%s'", val_split)

    return train_loader, val_loader, tokenizer
