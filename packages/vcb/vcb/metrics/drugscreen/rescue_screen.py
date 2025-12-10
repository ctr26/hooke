from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.task.drugscreen import add_compound_perturbation_to_obs
from vcb.metrics.drugscreen.hit_score import aggregate_hit_scores_per_compound, compute_hit_scores
from vcb.metrics.drugscreen.preprocessing import isolation_outlier_mask
from vcb.metrics.drugscreen.prometheus import embed_in_prometheus_space, prometheus_plot
from vcb.metrics.drugscreen.sampling import (
    aggregate_treatment_data,
    compute_glyph_sample_size,
    get_stratified_sample_counts,
    sample_and_aggregate,
)
from vcb.metrics.drugscreen.utils import SynchronizedDataset
from vcb.metrics.utils.transforms import pcaw_transform_data


def rescue_screen_analysis(
    data: AnnotatedDataMatrix,
    plot_destination: Path | None = None,
    plot_hit_threshold: float | None = None,
    plot_compounds: dict[str, list[str]] | None = None,
    n_standard_deviations_threshold: int = 10,
    random_state: int = 0,
    n_glyphs_for_clouds: int = 1000,
    embedding: Literal["pca"] | None = None,
    embedding_kwargs: dict | None = None,
):
    """
    Given drugscreen data, scores the ability of a compound to revert a disease state to a healthy state.
    """

    data = SynchronizedDataset(obs=data.obs.clone(), X=data.X.copy())

    if embedding_kwargs is None:
        embedding_kwargs = {}
    if "random_state" not in embedding_kwargs:
        embedding_kwargs["random_state"] = random_state

    # TODO (cwognum): Would love to incorporate TxAM here!
    if embedding == "pca":
        logger.info(f"Embedding the data using PCA with {embedding_kwargs}...")
        data.X = PCA(**embedding_kwargs).fit_transform(data.X)
    elif embedding is not None:
        raise ValueError(f"Unsupported embedding: {embedding}")

    # Step 0: We do the analysis per experiment.
    for (experiment,), experiment_data in tqdm(
        data.group_by("experiment_label"),
        desc="Performing rescue screen analysis per experiment",
        total=data.obs["experiment_label"].n_unique(),
        leave=False,
    ):
        # Step 1: Determine the glyph sample size
        glyph_sample_size = compute_glyph_sample_size(experiment_data.obs)

        # Step 2: Remove outliers from the control and disease model.
        control = experiment_data.filter(pl.col("is_negative_control"))
        if len(control) == 0:
            raise RuntimeError(
                f"No control data found for experiment {experiment}. "
                "Did you include the negative controls as part of the predictions?"
            )
        control.filter(isolation_outlier_mask(control.X, random_state=random_state))

        disease_model = experiment_data.filter(pl.col("is_base_state"))
        if len(disease_model) == 0:
            raise RuntimeError(
                f"No disease model data found for experiment {experiment}. "
                "Did you include the base states as part of the predictions?"
            )
        disease_model.filter(isolation_outlier_mask(disease_model.X, random_state=random_state))

        # Step 3: Data transformation using PCA Whitening
        experiment_data = control.join(disease_model).join(experiment_data.filter(pl.col("drugscreen_query")))
        experiment_data = pcaw_transform_data(experiment_data, random_state=random_state)

        # Step 4: Embed in Prometheus space
        control = experiment_data.filter(pl.col("is_negative_control"))
        disease_model = experiment_data.filter(pl.col("is_base_state"))
        drugscreen = experiment_data.filter(pl.col("drugscreen_query"))

        if len(drugscreen) == 0:
            # Because we subset the dataset to the test set, therecan be experiments for which we have no perturbed observations.
            # The group_by will still iterate over these as they occur in the base states and controls.
            logger.info(
                f"No drugscreen data found for experiment {experiment}. "
                "This is expected for some experiments."
            )
            continue

        control_center = control.X.mean(axis=0)
        disease_model_center = disease_model.X.mean(axis=0)

        control.X = embed_in_prometheus_space(control.X, control_center, disease_model_center)
        disease_model.X = embed_in_prometheus_space(disease_model.X, control_center, disease_model_center)
        drugscreen.X = embed_in_prometheus_space(drugscreen.X, control_center, disease_model_center)

        # Step 5: Aggregate treatment data
        drugscreen_xy, drugscreen_labels = aggregate_treatment_data(drugscreen, ["inchikey", "concentration"])
        sorted_indices = sorted(range(len(drugscreen_labels)), key=lambda i: drugscreen_labels[i])
        drugscreen_xy = drugscreen_xy[sorted_indices]
        drugscreen_labels = [drugscreen_labels[i] for i in sorted_indices]

        # Step 6: Sample and aggregate the disease and healthy clouds
        classes = drugscreen.obs[["inchikey", "concentration"]].rows()
        dt = np.dtype([("inchikey", "U27"), ("concentration", float)])
        classes = np.array(classes, dtype=dt)

        sample_sizes = get_stratified_sample_counts(
            classes,
            n_glyphs_for_clouds,
            rng=np.random.RandomState(random_state),
        )

        control_xy = sample_and_aggregate(
            control,
            drugscreen,
            ["inchikey", "concentration"],
            sample_sizes,
            glyph_sample_size,
            random_state=random_state,
        )

        disease_model_xy = sample_and_aggregate(
            disease_model,
            drugscreen,
            ["inchikey", "concentration"],
            sample_sizes,
            glyph_sample_size,
            random_state=random_state,
        )

        # Step 7: Compute the hit scores
        perturbation_level_hit_scores = compute_hit_scores(
            healthy_cloud=control_xy,
            disease_model_cloud=disease_model_xy,
            treatment_data=drugscreen_xy,
            n_sigma_y=n_standard_deviations_threshold,
            verbose=True,
        )

        compound_level_hit_scores = aggregate_hit_scores_per_compound(  # noqa: F841
            perturbation_scores=perturbation_level_hit_scores,
            perturbation_labels=drugscreen_labels,
        )

        # Step 8: Plot (Optional)
        if plot_destination is not None:
            plot_destination.mkdir(parents=True, exist_ok=True)

            drugscreen_xy_plot = drugscreen_xy
            drugscreen_labels_plot = drugscreen_labels

            # Filter the compounds to show in the plot.
            if plot_compounds is not None:
                indices = [
                    idx
                    for idx, (inchikey, _) in enumerate(drugscreen_labels)
                    if inchikey in plot_compounds[experiment]
                ]
                drugscreen_xy_plot = drugscreen_xy_plot[indices]
                drugscreen_labels_plot = [drugscreen_labels[i] for i in indices]

                if plot_hit_threshold is not None:
                    logger.warning(
                        "plot_compounds takes precedence over plot_hit_threshold. Ignoring plot_hit_threshold."
                    )

                expected = set(plot_compounds[experiment])
                found = set([inchikey for inchikey, _ in drugscreen_labels_plot])
                if expected != found:
                    logger.warning(
                        f"Not all plot_compounds are present in the drugscreen labels. "
                        f"Expected: {len(expected)}, Found: {len(found)}. Difference: {expected - found}"
                    )

            elif plot_hit_threshold is not None:
                indices = [
                    idx
                    for idx, (inchikey, _) in enumerate(drugscreen_labels)
                    if compound_level_hit_scores.get(inchikey, -1) >= plot_hit_threshold
                ]
                drugscreen_xy_plot = drugscreen_xy[indices]
                drugscreen_labels_plot = [drugscreen_labels[i] for i in indices]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax = prometheus_plot(
                disease_model_xy=disease_model_xy,
                control_xy=control_xy,
                drugscreen_xy=drugscreen_xy_plot,
                drugscreen_labels=drugscreen_labels_plot,
                n_standard_deviations_threshold=n_standard_deviations_threshold,
                ax=ax,
            )
            title = f"Prometheus Plot for Experiment {experiment}"
            if plot_compounds is not None:
                selected = set([inchikey for inchikey, _ in drugscreen_labels_plot])
                title += f" (Plotting a subset of {len(selected)} compounds)"
            elif plot_hit_threshold is not None:
                title += f" (Plotting compounds with hit score >= {plot_hit_threshold})"
            ax.set_title(title)

            fig.tight_layout()
            fig.savefig(plot_destination / f"rescue_screen_{experiment}.jpg")
            plt.close(fig)

        # Step 9: Return the hit scores
        yield (
            experiment,
            compound_level_hit_scores,
            perturbation_level_hit_scores,
            [inchikey for inchikey, _ in drugscreen_labels_plot],
        )


if __name__ == "__main__":
    # NOTE (cwognum): If you're on BioHive, this is a nice test dataset.
    #   It's the same example used in the RXRX Bridge Course on Hit Scoring.
    #   Specifically, it's experiment `VHL-Core01-H-C700a`.
    #   Especially the compounds `REC-0064744` & `REC-0001788` are interesting.
    path = "/rxrx/data/valence/internal_benchmarking/context_vcds1/dart_example__v1_0"
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=path).model_dump())
    dataset.obs = add_compound_perturbation_to_obs(dataset.obs)
    next(rescue_screen_analysis(dataset, plot_destination=Path("."), plot_hit_threshold=0.75))
