from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from scipy import linalg
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.task.singles import add_single_perturbation_to_obs
from vcb.metrics.drugscreen.utils import SynchronizedDataset, stack
from vcb.metrics.utils.transforms import pcaw_transform_data


def tvn_on_controls(
    data: SynchronizedDataset, groupby_columns: list[str] | None = None
) -> SynchronizedDataset:
    """
    Apply TVN (Typical Variation Normalization) to the data based on the control perturbation units.
    Note that the data is first centered and scaled based on the control units.

    NOTE (cwognum): This function frequently changes the dtype to np.complex128, the complex part of
        which we simply ignore. Is that something we need to look into?
    """

    if groupby_columns is None:
        groupby_columns = ["batch_center"]

    control = data.filter(pl.col("is_negative_control"))
    target_cov = np.cov(control.X, rowvar=False, ddof=1) + 0.5 * np.eye(data.X.shape[1])

    collect = []
    for _, group in tqdm(
        data.group_by(groupby_columns), desc="TVN", total=data.obs[groupby_columns].n_unique()
    ):
        group_control = group.filter(pl.col("is_negative_control"))
        source_cov = np.cov(group_control.X, rowvar=False, ddof=1) + 0.5 * np.eye(data.X.shape[1])
        group.X = np.matmul(group.X, linalg.fractional_matrix_power(source_cov, -0.5))
        group.X = np.matmul(group.X, linalg.fractional_matrix_power(target_cov, 0.5))
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


def map_building_pipeline(
    data: AnnotatedDataMatrix,
    perturbation_groupby_columns: list[str],
    plot_destination: Path | None = None,
    cell_type_subset: list[str] | None = None,
    perturbation_order: list[str] | None = None,
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

        group = pcaw_transform_data(group, ["batch_center"])
        group = tvn_on_controls(group)

        embedding_per_perturbation = aggregate(group, perturbation_groupby_columns)
        perturbations = sorted(embedding_per_perturbation.keys())
        embeddings = np.vstack([embedding_per_perturbation[pert] for pert in perturbations])

        if len(embeddings) < 2:
            logger.warning(
                f"After all transformations, we're left with just {len(embeddings)} perturbations "
                f"for cell type {cell_type}, skipping."
            )
            continue

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

        if plot_destination is not None:
            plot_destination.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(12, 10))
            cmap = sns.color_palette("vlag", as_cmap=True)
            sns.heatmap(mapmat, ax=ax, cmap=cmap, vmin=-1, vmax=1)

            prettified_labels = [", ".join(p) for p in perturbations]
            ax.set_xticklabels(prettified_labels, rotation=90)
            ax.set_yticklabels(prettified_labels, rotation=0)
            ax.set_title(f"Cosine Similarity Map for Cell Type {cell_type}")

            fig.tight_layout()
            fig.savefig(plot_destination / f"map_{cell_type}.jpg")
            plt.close(fig)

        yield mapmat, cell_type, perturbations


if __name__ == "__main__":
    # Here's an example of how to run the pipeline in a stand-alone manner,
    # without the need for all other code in VCB.

    path = "/rxrx/data/valence/internal_benchmarking/vcds1/cross_cell_line_trek__v1_1"
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=path).model_dump())

    indices = (
        dataset.obs.with_row_index("original_index")
        .filter(~pl.col("cell_type").eq("HUVEC"))
        .filter((pl.col("perturbations").list.len().le(1)))["original_index"]
        .to_list()
    )
    dataset.filter(obs_indices=indices)
    dataset.obs = add_single_perturbation_to_obs(dataset.obs)

    for _ in map_building_pipeline(
        dataset,
        perturbation_groupby_columns=["inchikey"],
        plot_destination=Path("."),
    ):
        pass
