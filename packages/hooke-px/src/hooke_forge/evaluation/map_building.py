"""EFAAR map-building pipeline.

Extracted from VCB (vcb.metrics.map_building) into a self-contained module.
Builds perturbation similarity maps from phenomics embeddings using:
  PCA whitening -> TVN (Typical Variation Normalization) -> aggregation -> cosine similarity.

Required obs columns: cell_type, is_negative_control, batch_center,
plus perturbation columns (e.g. inchikey).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SynchronizedDataset
# ---------------------------------------------------------------------------


class SynchronizedDataset(BaseModel):
    """Keeps metadata (obs) and feature matrix (X) in sync."""

    obs: pl.DataFrame
    X: np.ndarray

    _index_column: str = "original_index"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_same_length(self) -> "SynchronizedDataset":
        if len(self.obs) != self.X.shape[0]:
            raise ValueError(
                f"obs and X length mismatch: {len(self.obs)} != {self.X.shape[0]}"
            )
        if self._index_column in self.obs.columns:
            self.obs = self.obs.drop(self._index_column)
        self.obs = self.obs.with_row_index(self._index_column)
        return self

    def filter(self, predicate: pl.Expr) -> "SynchronizedDataset":
        obs = self.obs.filter(predicate)
        indices = obs[self._index_column].to_list()
        return SynchronizedDataset(obs=obs, X=self.X[indices])

    def join(self, other: "SynchronizedDataset") -> "SynchronizedDataset":
        obs = pl.concat([self.obs, other.obs])
        X = np.vstack([self.X, other.X])
        return SynchronizedDataset(obs=obs, X=X)

    def group_by(self, groupby_columns: list[str]):
        for name, group in self.obs.group_by(groupby_columns, maintain_order=True):
            indices = group[self._index_column].to_list()
            yield name, SynchronizedDataset(obs=group, X=self.X[indices])

    def __len__(self) -> int:
        return len(self.obs)


def stack(datasets: list[SynchronizedDataset]) -> SynchronizedDataset:
    """Concatenate a list of SynchronizedDatasets."""
    if len(datasets) == 1:
        return datasets[0]
    data = datasets[0]
    for dataset in datasets[1:]:
        data = data.join(dataset)
    return data


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


class Map(BaseModel):
    """Perturbation similarity map with I/O."""

    similarity_matrix: np.ndarray
    embeddings: np.ndarray
    perturbations: list[tuple[str, ...]]
    cell_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load(cls, path: Path) -> "Map":
        cache = np.load(path)
        return cls(
            similarity_matrix=cache["similarity_matrix"],
            embeddings=cache["embeddings"],
            perturbations=cache["perturbations"].tolist(),
            cell_type=str(cache["cell_type"]),
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            similarity_matrix=self.similarity_matrix,
            embeddings=self.embeddings,
            perturbations=np.array(self.perturbations),
            cell_type=self.cell_type,
        )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def center_scale_transform(
    data: SynchronizedDataset, groupby_columns: list[str]
) -> SynchronizedDataset:
    """Center and scale per group, fitting on negative controls."""
    collect = []
    for name, group in data.group_by(groupby_columns):
        fit_data = group.filter(pl.col("is_negative_control"))
        if len(fit_data) == 0:
            log.warning(f"No negative controls in group {name}, skipping scaling")
            continue
        scaler = StandardScaler()
        scaler.fit(fit_data.X)
        group.X = scaler.transform(group.X)
        collect.append(group)
    return stack(collect)


def pca_transform(
    data: SynchronizedDataset,
    n_components: int | float = 0.999,
    random_state: int | None = None,
) -> SynchronizedDataset:
    """PCA transform fitted on negative controls."""
    fit_data = data.filter(pl.col("is_negative_control"))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(fit_data.X)
    data.X = pca.transform(data.X)
    return data


def pcaw_transform_data(
    data: SynchronizedDataset,
    groupby_columns: list[str] | None = None,
    random_state: int | None = None,
) -> SynchronizedDataset:
    """PCA Whitening: center-scale -> PCA -> center-scale."""
    if groupby_columns is None:
        groupby_columns = ["batch_center"]
    data = center_scale_transform(data, groupby_columns)
    data = pca_transform(data, random_state=random_state)
    data = center_scale_transform(data, groupby_columns)
    return data


# ---------------------------------------------------------------------------
# TVN + aggregation + map building
# ---------------------------------------------------------------------------


def matrix_power_eigh(mat: np.ndarray, power: float) -> np.ndarray:
    """Fractional matrix power via eigendecomposition (symmetric PSD matrices)."""
    eigenvals, eigenvecs = np.linalg.eigh(mat)
    eigenvals = np.maximum(eigenvals, 1e-10)
    return eigenvecs @ np.diag(np.power(eigenvals, power)) @ eigenvecs.T


def tvn_on_controls(
    data: SynchronizedDataset,
    groupby_columns: list[str] | None = None,
) -> SynchronizedDataset:
    """Typical Variation Normalization based on negative controls."""
    if groupby_columns is None:
        groupby_columns = ["batch_center"]

    control = data.filter(pl.col("is_negative_control"))
    target_cov = np.cov(control.X, rowvar=False, ddof=1) + 0.5 * np.eye(
        data.X.shape[1]
    )
    target_cov_half = matrix_power_eigh(target_cov, 0.5)

    collect = []
    for _, group in tqdm(
        data.group_by(groupby_columns),
        desc="TVN",
        total=data.obs[groupby_columns].n_unique(),
    ):
        group_control = group.filter(pl.col("is_negative_control"))
        source_cov = np.cov(group_control.X, rowvar=False, ddof=1) + 0.5 * np.eye(
            data.X.shape[1]
        )
        source_cov_half_inv = matrix_power_eigh(source_cov, -0.5)
        group.X = group.X @ source_cov_half_inv @ target_cov_half
        collect.append(group)

    return stack(collect)


def aggregate(
    data: SynchronizedDataset,
    perturbation_groupby_columns: list[str],
) -> dict[str, np.ndarray]:
    """Mean-aggregate replicate embeddings per perturbation (excluding controls)."""
    data = data.filter(~pl.col("is_negative_control"))

    n = len(data)
    mask = pl.lit(True)
    for column in perturbation_groupby_columns:
        mask = mask & pl.col(column).is_not_null()
    data = data.filter(mask)

    if len(data) != n:
        diff = n - len(data)
        pct = (diff / n) * 100
        log.warning(
            f"Filtered out {diff} rows ({pct:.1f}%) with null perturbation values"
        )

    collect = {}
    for perturbation, group in data.group_by(perturbation_groupby_columns):
        collect[perturbation] = np.mean(group.X, axis=0)
    return collect


def build_map(
    data: SynchronizedDataset,
    perturbation_groupby_columns: list[str],
    cell_type: str,
    perturbation_order: list[str] | None = None,
) -> Map | None:
    """Build a perturbation similarity map using the EFAAR pipeline."""
    data = pcaw_transform_data(data, ["batch_center"])
    data = tvn_on_controls(data)

    embedding_per_perturbation = aggregate(data, perturbation_groupby_columns)
    perturbations = sorted(embedding_per_perturbation.keys())
    embeddings = np.vstack([embedding_per_perturbation[p] for p in perturbations])

    if len(embeddings) < 2:
        log.warning(
            f"Only {len(embeddings)} perturbations for cell type {cell_type}, skipping"
        )
        return None

    mat = pdist(embeddings, metric="cosine")
    square_mat = squareform(mat)

    if perturbation_order is not None:
        order = [
            perturbations.index(p)
            for p in perturbation_order
            if p in perturbations
        ]
    else:
        Z = linkage(mat, method="ward")
        order = leaves_list(Z)

    reordered_mat = square_mat[order, :][:, order]
    perturbations = [perturbations[i] for i in order]
    similarity = 1 - reordered_mat

    return Map(
        similarity_matrix=similarity,
        embeddings=embeddings,
        perturbations=perturbations,
        cell_type=cell_type,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def map_building_pipeline(
    data: SynchronizedDataset,
    perturbation_groupby_columns: list[str],
    save_destination: Path | None = None,
    cell_type_subset: list[str] | None = None,
    perturbation_order: list[str] | None = None,
    cache_dir: Path | None = None,
):
    """Build maps per cell type. Yields Map objects."""
    if cell_type_subset is not None:
        data = data.filter(pl.col("cell_type").is_in(cell_type_subset))
    else:
        cell_type_subset = data.obs["cell_type"].unique().to_list()

    log.info(f"Computing maps for cell types: {cell_type_subset}")

    for idx, ((cell_type,), group) in enumerate(data.group_by(["cell_type"])):
        log.info(
            f"Building map for {cell_type} "
            f"({idx + 1}/{data.obs['cell_type'].n_unique()})"
        )

        cache_path = cache_dir / f"map_{cell_type}.npz" if cache_dir else None

        if cache_path is not None and cache_path.exists():
            log.info(f"Loading from cache: {cache_path}")
            vmap = Map.load(cache_path)
        else:
            vmap = build_map(
                group, perturbation_groupby_columns, cell_type, perturbation_order
            )
            if cache_path is not None and vmap is not None:
                vmap.save(cache_path)

        if vmap is None:
            continue

        if save_destination is not None:
            _save_map_and_plot(vmap, save_destination, cell_type)

        yield vmap


def _save_map_and_plot(vmap: Map, save_destination: Path, cell_type: str):
    """Save map .npz and heatmap .jpg."""
    map_dir = save_destination / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    vmap.save(map_dir / f"map_{cell_type}.npz")

    plot_dir = save_destination / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(
        vmap.similarity_matrix,
        ax=ax,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        annot=len(vmap.perturbations) < 25,
    )

    if len(vmap.perturbations) == len(ax.get_xticks()):
        labels = [", ".join(p) for p in vmap.perturbations]
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_axis_off()

    ax.set_title(f"Cosine Similarity Map for Cell Type {cell_type}")
    fig.tight_layout()
    dpi = max(100, min(600, vmap.similarity_matrix.shape[0] // 10))
    fig.savefig(plot_dir / f"map_{cell_type}.jpg", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: build maps from inference output
# ---------------------------------------------------------------------------


def build_maps_from_inference(
    output_dir: Path,
    perturbation_cols: list[str] | None = None,
    batch_col: str = "batch_center",
    control_col: str = "is_negative_control",
    save_destination: Path | None = None,
    cell_type_subset: list[str] | None = None,
) -> list[Map]:
    """Build maps from inference output directory.

    Loads pred_phenom.zarr + prepared_metadata.parquet and runs the pipeline.

    Args:
        output_dir: Inference output directory containing features/ and prepared_metadata.parquet
        perturbation_cols: Columns to group perturbations by (default: ["inchikey"])
        batch_col: Column for batch grouping (default: "batch_center")
        control_col: Column indicating negative controls (default: "is_negative_control")
        save_destination: Where to save maps and plots (default: output_dir / "maps")
        cell_type_subset: Optional subset of cell types to process

    Returns:
        List of Map objects
    """
    import zarr

    if perturbation_cols is None:
        perturbation_cols = ["inchikey"]

    output_dir = Path(output_dir)
    if save_destination is None:
        save_destination = output_dir / "maps"

    # Load features
    zarr_path = output_dir / "features" / "pred_phenom.zarr"
    X = np.array(zarr.open(str(zarr_path), mode="r"))

    # If multiple samples, mean-aggregate across samples
    if X.ndim == 3:
        X = X.mean(axis=1)

    # Load metadata
    metadata_path = output_dir / "prepared_metadata.parquet"
    obs = pl.read_parquet(metadata_path)

    # Validate required columns
    required = ["cell_type", control_col, batch_col] + perturbation_cols
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in metadata: {missing}. "
            f"Available: {obs.columns}"
        )

    log.info(f"Loaded {len(obs)} observations with {X.shape[1]} features")

    data = SynchronizedDataset(obs=obs, X=X)

    maps = list(
        map_building_pipeline(
            data,
            perturbation_groupby_columns=perturbation_cols,
            save_destination=save_destination,
            cell_type_subset=cell_type_subset,
        )
    )

    log.info(f"Built {len(maps)} maps")
    return maps
