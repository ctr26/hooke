from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.task.drugscreen import add_compound_perturbation_to_obs
from vcb.metrics.drugscreen.hit_score import aggregate_hit_scores_per_compound, compute_hit_scores
from vcb.metrics.drugscreen.preprocessing import isolation_outlier_mask, pcaw_transform_data
from vcb.metrics.drugscreen.prometheus import embed_in_prometheus_space, prometheus_plot
from vcb.metrics.drugscreen.sampling import (
    aggregate_treatment_data,
    compute_glyph_sample_size,
    get_stratified_sample_counts,
    sample_and_aggregate,
)
from vcb.metrics.drugscreen.utils import SynchronizedDataset


def rescue_screen_analysis(
    data: AnnotatedDataMatrix,
    plot_destination: Path | None = None,
    plot_hit_threshold: float | None = None,
    n_standard_deviations_threshold: int = 10,
    random_state: int = 0,
    n_glyphs_for_clouds: int = 1000,
):
    """
    Given drugscreen data, scores the ability of a compound to revert a disease state to a healthy state.
    """

    data = SynchronizedDataset(obs=data.obs, X=data.X)

    # Step 0: We do the analysis per experiment.
    for (experiment,), experiment_data in data.group_by("experiment_label"):
        logger.info(f"Using experiment: {experiment}. Selected {len(experiment_data.obs)} observations.")

        # Step 1: Determine the glyph sample size
        glyph_sample_size = compute_glyph_sample_size(experiment_data.obs)

        # Step 2: Remove outliers from the control and disease model.
        control = experiment_data.filter(pl.col("is_negative_control"))
        control.filter(isolation_outlier_mask(control.X))

        disease_model = experiment_data.filter(pl.col("is_base_state"))
        disease_model.filter(isolation_outlier_mask(disease_model.X))

        # Step 3: Data transformation using PCA Whitening
        experiment_data = control.join(disease_model).join(experiment_data.filter(pl.col("drugscreen_query")))
        experiment_data = pcaw_transform_data(experiment_data)

        # Step 4: Embed in Prometheus space
        control = experiment_data.filter(pl.col("is_negative_control"))
        disease_model = experiment_data.filter(pl.col("is_base_state"))
        drugscreen = experiment_data.filter(pl.col("drugscreen_query"))

        control_center = control.X.mean(axis=0)
        disease_model_center = disease_model.X.mean(axis=0)

        control.X = embed_in_prometheus_space(control.X, control_center, disease_model_center)
        disease_model.X = embed_in_prometheus_space(disease_model.X, control_center, disease_model_center)

        drugscreen.obs = add_compound_perturbation_to_obs(drugscreen.obs)
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

            if plot_hit_threshold is not None:
                # Filter the compounds to show in the plot.
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

            fig.tight_layout()
            fig.savefig(plot_destination / f"rescue_screen_{experiment}.jpg")

        # Step 9: Return the hit scores
        yield experiment, compound_level_hit_scores, perturbation_level_hit_scores


if __name__ == "__main__":
    # NOTE (cwognum): If you're on BioHive, this is a nice test dataset.
    #   It's the same example used in the RXRX Bridge Course on Hit Scoring.
    #   Specifically, it's experiment `VHL-Core01-H-C700a`.
    #   Especially the compounds `REC-0064744` & `REC-0001788` are interesting.
    path = "/rxrx/data/valence/internal_benchmarking/context_vcds1/dart_example__v1_0"
    dataset = AnnotatedDataMatrix(**DatasetDirectory(root=path).model_dump())
    next(rescue_screen_analysis(dataset, plot_destination=Path("."), plot_hit_threshold=0.75))
