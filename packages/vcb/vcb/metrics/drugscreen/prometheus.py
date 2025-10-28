import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from vcb.metrics.drugscreen.hit_score import compute_hit_scores


def embed_in_prometheus_space(
    features: np.ndarray, p0_vector: np.ndarray, p1_vector: np.ndarray
) -> np.ndarray:
    """Transform every row of a numpy array of observations into prometheus xy.

    Given three vectors, data D, p0_vector C, and p1_vector P
    computes a projection and a rejection for the vector D.

    We will first recenter the data setting the origin at C.
    So D := D-C and P:=P-C  Then the projection is defined as the
    projection of D onto P divided by the norm of P. The rejection is
    the rejection of D onto P divided by the norm of P.
    """

    p0_vector = np.expand_dims(p0_vector, axis=0)
    p1_vector = np.expand_dims(p1_vector, axis=0)

    features_v = features - p0_vector
    principle_axis = p1_vector - p0_vector

    # Length of the projection of the rows of data onto p1_vector
    projection = np.dot(features_v, principle_axis.T) / np.linalg.norm(principle_axis)

    # Distance of every observation from the origin (p0_vector)
    # This will be the hypotenuse for our right triangle.
    features_norm = np.expand_dims(np.linalg.norm(features_v, axis=1), axis=1)

    # Pythagorean Theorem
    rejection = np.sqrt(features_norm**2 - projection**2)

    # Scale such that the projection of the principle axis is 1
    scaling_factor = np.linalg.norm(principle_axis)
    projection_scaled = projection / scaling_factor
    rejection_scaled = rejection / scaling_factor

    xy = np.concatenate([projection_scaled, rejection_scaled], axis=1)
    logger.debug(f"Embedded in Prometheus space: {features.shape} -> {xy.shape}")

    return xy


def prometheus_plot(
    disease_model_xy: np.ndarray,
    control_xy: np.ndarray,
    drugscreen_xy: np.ndarray,
    drugscreen_labels: list[str],
    n_standard_deviations_threshold: int = 10,
    ax: plt.Axes | None = None,
):
    """Visualize the Prometheus plot.

    This is a scatter plot of the disease model and control data, with the drugscreen data overlaid.
    The hit scores are computed and used to shade the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Plot the disease model and healthy data clouds.
    ax = sns.scatterplot(x=disease_model_xy[:, 0], y=disease_model_xy[:, 1], alpha=0.25, color="red")
    ax = sns.scatterplot(x=control_xy[:, 0], y=control_xy[:, 1], alpha=0.25, color="green", ax=ax)

    # Plot treatment data.
    # For each compound, this is a "curve" of points corresponding to different doses.
    hue_labels = np.array([label[0] for label in drugscreen_labels])
    unique_labels = np.unique(hue_labels)
    for label in unique_labels:
        mask = hue_labels == label
        ax.plot(
            drugscreen_xy[mask, 0],
            drugscreen_xy[mask, 1],
            marker="o",
            label=label,
            alpha=0.5,
        )

    # Determine y-axis limits
    y_std = max(disease_model_xy[:, 1].std(), control_xy[:, 1].std())
    y_mean = max(disease_model_xy[:, 1].mean(), control_xy[:, 1].mean())
    y_min = y_mean - n_standard_deviations_threshold * y_std

    # Draw the hit score contours
    # +1 here is used to create some padding around the data
    # Otherwise the contours wouldn't be fully visible.
    y_max = y_mean + (n_standard_deviations_threshold + 1) * y_std
    x = np.arange(-1, 2, 3 / 100)
    y = np.arange(y_min, y_max, (y_max - y_min) / 100)
    grid = np.meshgrid(x, y)
    xi = grid[0].flatten()
    yi = grid[1].flatten()
    z = compute_hit_scores(
        healthy_cloud=control_xy,
        disease_model_cloud=disease_model_xy,
        treatment_data=np.vstack([xi, yi]).T,
        n_sigma_y=n_standard_deviations_threshold,
        verbose=False,
    )

    levels = [0, 0.25, 0.5, 0.75, 1.0]
    z = z.reshape(grid[0].shape)
    ax.contour(grid[0], grid[1], z, levels=levels, colors="black", linestyles="dashed", alpha=0.1)
    ax.contourf(grid[0], grid[1], z, levels=levels, cmap="Greens", alpha=0.1)

    # Basic labeling and styling.
    ax.grid()
    ax.set_ylabel("Side Effect Score Median", fontsize=12, fontweight="bold")
    ax.set_xlabel("Disease Score Median", fontsize=12, fontweight="bold")
    ax.set_title("Prometheus Plot", fontsize=14, fontweight="bold")

    # Set the axis limits.
    # We set the limits to twice the standard deviation to create some padding
    y_max = y_mean + n_standard_deviations_threshold * 2 * y_std
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1, 2)
    return ax
