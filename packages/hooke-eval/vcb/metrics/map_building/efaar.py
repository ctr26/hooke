from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.task.singles import add_single_perturbation_to_obs
from vcb.metrics.drugscreen.utils import SynchronizedDataset, stack
from vcb.metrics.map_building.map import Map
from vcb.metrics.utils.transforms import pcaw_transform_data


def matrix_power_eigh(mat: np.ndarray, power: float) -> np.ndarray:
    """
    Use eigendecomposition to compute the square root of the matrix.
    This is much faster and more stable than using np.linalg.fractional_matrix_power,
    but it does require the matrix to be symmetric and positive semi-definite.
    In this specific flow, that should always be the case.
    """
    # 1. Eigendecomposition (A = Q @ Lambda @ Q.T)
    eigenvals, eigenvecs = np.linalg.eigh(mat)

    # 2. Avoid numerical issues and handle non-positive eigenvalues
    #    For real symmetric positive semi-definite matrices, eigenvalues should be >= 0.
    #    We can set small, near-zero eigenvalues to a minimum positive value
    #    before applying the power. This is crucial for inverse powers (p < 0).
    eigenvals = np.maximum(eigenvals, 1e-10)

    # 3. Reconstruction (A^p = Q @ Lambda^p @ Q.T)
    return eigenvecs @ np.diag(np.power(eigenvals, power)) @ eigenvecs.T


def tvn_on_controls(
    data: SynchronizedDataset, groupby_columns: list[str] | None = None, use_eigendecomposition: bool = True
) -> SynchronizedDataset:
    """
    Apply TVN (Typical Variation Normalization) to the data based on the control perturbation units.
    Note that the data is first centered and scaled based on the control units.
    """

    if groupby_columns is None:
        groupby_columns = ["batch_center"]

    control = data.filter(pl.col("is_negative_control"))
    target_cov = np.cov(control.X, rowvar=False, ddof=1) + 0.5 * np.eye(data.X.shape[1])

    if use_eigendecomposition:
        target_cov_half = matrix_power_eigh(target_cov, 0.5)
    else:
        target_cov_half = np.linalg.fractional_matrix_power(target_cov, 0.5)

    collect = []
    for _, group in tqdm(
        data.group_by(groupby_columns), desc="TVN", total=data.obs[groupby_columns].n_unique()
    ):
        group_control = group.filter(pl.col("is_negative_control"))
        source_cov = np.cov(group_control.X, rowvar=False, ddof=1) + 0.5 * np.eye(data.X.shape[1])

        if use_eigendecomposition:
            source_cov_half_inv = matrix_power_eigh(source_cov, -0.5)
        else:
            source_cov_half_inv = np.linalg.fractional_matrix_power(source_cov, -0.5)

        group.X = group.X @ source_cov_half_inv @ target_cov_half
        collect.append(group)

    return stack(collect)


def aggregate(
    data: SynchronizedDataset,
    perturbation_groupby_columns: list[str],
) -> dict[str, np.ndarray]:
    """
    Apply the mean or median aggregation to replicate embeddings for each perturbation.
    """
    # Filter out any negative controls.
    data = data.filter(~pl.col("is_negative_control"))

    n = len(data)
    mask = pl.lit(True)
    for column in perturbation_groupby_columns:
        mask = mask & pl.col(column).is_not_null()
    data = data.filter(mask)

    if len(data) != n:
        diff = n - len(data)
        pct = (diff / n) * 100
        logger.warning(
            f"Filtered out {diff} rows ({pct:.1f}%) from the data because the perturbation was not specified."
        )

    collect = {}
    for perturbation, group in data.group_by(perturbation_groupby_columns):
        mean_embedding = np.mean(group.X, axis=0)
        collect[perturbation] = mean_embedding

    return collect


def build_map(
    data: SynchronizedDataset,
    perturbation_groupby_columns: list[str],
    cell_type: str,
    perturbation_order: list[str] | None = None,
) -> Map:
    """
    Build the map using the EFAAR pipeline.
    """

    data = pcaw_transform_data(data, ["batch_center"])
    data = tvn_on_controls(data)

    embedding_per_perturbation = aggregate(data, perturbation_groupby_columns)
    perturbations = sorted(embedding_per_perturbation.keys())
    embeddings = np.vstack([embedding_per_perturbation[pert] for pert in perturbations])

    if len(embeddings) < 2:
        logger.warning(
            f"After all transformations, we're left with just {len(embeddings)} perturbations "
            f"for cell type {cell_type}, skipping."
        )
        return None

    mat = pdist(embeddings, metric="cosine")
    square_mat = squareform(mat)

    if perturbation_order is not None:
        order = [perturbations.index(pert) for pert in perturbation_order]
    else:
        # Reorder such that similar perturbations are close together and close to the diagonal.
        Z = linkage(mat, method="ward")
        order = leaves_list(Z)

    reordered_mat = square_mat[order, :]
    reordered_mat = reordered_mat[:, order]
    perturbations = [perturbations[i] for i in order]

    # Go from distance [0, 2] to similarity [-1, 1].
    mapmat = 1 - reordered_mat

    return Map(
        similarity_matrix=mapmat,
        embeddings=embeddings,
        perturbations=perturbations,
        cell_type=cell_type,
    )


def map_building_pipeline(
    data: AnnotatedDataMatrix,
    perturbation_groupby_columns: list[str],
    save_destination: Path | None = None,
    cell_type_subset: list[str] | None = None,
    perturbation_order: list[str] | None = None,
    cache_dir: Path | None = None,
):
    """
    Build the map using the EFAAR pipeline.
    """
    data = SynchronizedDataset(obs=data.obs.clone(), X=data.X.copy())

    if cell_type_subset is not None:
        data = data.filter(pl.col("cell_type").is_in(cell_type_subset))
    else:
        cell_type_subset = data.obs["cell_type"].unique().to_list()

    logger.info(f"Computing the map for the following cell types: {cell_type_subset}")

    for idx, ((cell_type,), group) in enumerate(data.group_by(["cell_type"])):
        logger.info(
            f"Creating a map for cell type {cell_type} ({idx + 1} / {data.obs['cell_type'].n_unique()})"
        )

        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"map_{cell_type}.npz"

        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading map from cache: {cache_path}")
            vmap = Map.load(cache_path)
        else:
            vmap = build_map(group, perturbation_groupby_columns, cell_type, perturbation_order)
            if cache_path is not None:
                vmap.save(cache_path)

        # If still None, there weren't enough perturbations to build a map.
        if vmap is None:
            continue

        if save_destination is not None:
            map_destination = save_destination / "maps"
            map_destination.mkdir(parents=True, exist_ok=True)
            vmap.save(map_destination / f"map_{cell_type}.npz")

            plot_destination = save_destination / "plots"
            plot_destination.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Fun fact: "vlag" is the Dutch word for "flag", which, like the color palette, is red-white-blue.
            # Aren't you happy about that tidbit of information? No? Oh... Okay, moving on...
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
                # Enough space to show all labels.
                prettified_labels = [", ".join(p) for p in vmap.perturbations]
                ax.set_xticklabels(prettified_labels, rotation=90)
                ax.set_yticklabels(prettified_labels, rotation=0)
            else:
                # Too many labels. No need to show any.
                ax.set_axis_off()

            ax.set_title(f"Cosine Similarity Map for Cell Type {cell_type}")

            fig.tight_layout()
            dpi = max(100, min(600, vmap.similarity_matrix.shape[0] // 10))
            fig.savefig(plot_destination / f"map_{cell_type}.jpg", dpi=dpi)
            plt.close(fig)

        yield vmap


if __name__ == "__main__":
    # Here's an example of how to run the pipeline in a stand-alone manner,
    # without the need for all other code in VCB.

    path = "/rxrx/data/valence/internal_benchmarking/vcds1/cross_cell_line__brightfield__v1_1"
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=path).model_dump())
    dataset.obs = add_single_perturbation_to_obs(dataset.obs)

    for _ in map_building_pipeline(
        dataset,
        perturbation_groupby_columns=["inchikey"],
        plot_destination=Path("."),
    ):
        pass
