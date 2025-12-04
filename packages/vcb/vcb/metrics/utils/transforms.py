import polars as pl
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vcb.metrics.drugscreen.utils import SynchronizedDataset, stack


def center_scale_transform(data: SynchronizedDataset, groupby_columns: list[str]) -> SynchronizedDataset:
    """
    Center and scale the data.
    """

    logger.debug(
        f"Prior to the center scale transform, we have a mean of {data.X.mean(axis=0).mean():.2f} "
        f"and a variance of {data.X.var(axis=0).mean():.2f} across all features"
    )

    collect = []

    # We scale the data for each group separately.
    # This is done to correct for batch effects.
    for name, group in data.group_by(groupby_columns):
        # We fit the scaler to the negative controls.
        # We want the negative controls to be centered at the origin.
        fit_data = group.filter(pl.col("is_negative_control"))
        if len(fit_data) == 0:
            logger.warning(f"No negative controls found in group {name}. Skipping scaling!")
            continue

        zero_var_features = (fit_data.X.var(axis=0) == 0).sum()
        if zero_var_features > 0:
            logger.warning(
                f"Found {zero_var_features} features with zero variance in group {name}. "
                "These will not be scaled."
            )

        # Center scale the data.
        scaler = StandardScaler()
        scaler.fit(fit_data.X)

        group.X = scaler.transform(group.X)
        scaled_fit_data = scaler.transform(fit_data.X)

        logger.debug(
            f"After scaling the data in group {name}, the fitted data "
            f"has mean {scaled_fit_data.mean(axis=0).mean():.2f} "
            f"and variance {scaled_fit_data.var(axis=0).mean():.2f} across features. "
            f"The scaled data has mean {group.X.mean(axis=0).mean():.2f} "
            f"and variance {group.X.var(axis=0).mean():.2f} across features."
        )

        collect.append(group)

    # Stack the data and metadata back together.
    # Reset the index column now that metadata and data are aligned.
    data = stack(collect)
    return data


def pca_transform(data: SynchronizedDataset, n_components: int | float = 0.999) -> SynchronizedDataset:
    """
    PCA transform of the data.

    By using PCA, the covariance matrix is the identity matrix.
    Meaning the features are uncorrelated.
    """
    fit_data = data.filter(pl.col("is_negative_control"))
    if len(fit_data) == 0:
        raise RuntimeError("No negative controls found in group")

    pca = PCA(n_components=n_components)
    pca.fit(fit_data.X)

    n_samples = fit_data.X.shape[0]
    n_dim = fit_data.X.shape[1]
    max_dim = min(n_samples - 1, n_dim)
    if pca.n_components_ > max_dim:
        logger.warning(
            "Use of PCA is suspicious here, because there are more components than the max number of non-trivial components."
        )

    shape_before = data.X.shape
    data.X = pca.transform(data.X)

    logger.debug(
        f"PCA transformed a dataset of shape {shape_before} to a dataset of shape {data.X.shape}. "
        f"After PCA, we have a mean of {data.X.mean(axis=0).mean():.2f} "
        f"and a variance of {data.X.var(axis=0).mean():.2f} across all features"
    )
    return data


def pcaw_transform_data(
    data: SynchronizedDataset, groupby_columns: list[str] | None = None
) -> SynchronizedDataset:
    """
    Transform the data using PCA Whitening.
    """
    if groupby_columns is None:
        groupby_columns = ["batch_center"]

    data = center_scale_transform(data, groupby_columns)
    data = pca_transform(data)
    data = center_scale_transform(data, groupby_columns)
    return data
