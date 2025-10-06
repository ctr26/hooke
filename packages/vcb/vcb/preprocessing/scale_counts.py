import numpy as np
import polars as pl
from loguru import logger


class RawCountScaler:
    """
    A scaler for Transcriptomics data.

    This scaler will rescale the data to a desired library size and log transform the data.
    """

    def __init__(self, desired_library_size: int | None = None):
        """If you know the desired library size, you can directly set it at initialization."""
        self.desired_library_size = desired_library_size

    @property
    def fitted(self) -> bool:
        return self.desired_library_size is not None

    def fit(self, reference: np.ndarray):
        """If you don't know the desired library size, you can fit it from a reference array."""
        self.desired_library_size = np.median(np.sum(reference, axis=1))

    def transform(self, data: np.ndarray, is_log1p_transformed: bool = False):
        if not self.fitted:
            raise ValueError(
                "The RawCountScaler is not fitted. "
                "Please pass a `reference` array or set `desired_library_size`"
            )

        d = data.copy()

        # Reset log1p transformation
        if is_log1p_transformed:
            d = np.exp(d) - 1

        # Reset any prior library size scaling
        d = d / d.sum(axis=1, keepdims=True)

        # And then rescale and log transform the data
        d = d * self.desired_library_size
        d = np.log1p(d)

        changes = self._summarize_changes(data, d)
        logger.info(f"Rescaled the data to a library size of {self.desired_library_size}:\n{changes}")

        return d

    def fit_transform(
        self,
        data: np.ndarray,
        reference: np.ndarray | None = None,
        is_log1p_transformed: bool = False,
    ):
        if reference is not None:
            self.fit(reference)
        return self.transform(data, is_log1p_transformed)

    def _summarize_changes(self, before: np.ndarray, after: np.ndarray) -> pl.DataFrame:
        """Summarize the data distribution."""
        return pl.DataFrame(
            {
                "Metric": ["min", "max", "mean"],
                "Before": [before.min(), before.max(), before.mean()],
                "After": [after.min(), after.max(), after.mean()],
            },
            schema={"Metric": pl.Utf8, "Before": pl.Float64, "After": pl.Float64},
        )
