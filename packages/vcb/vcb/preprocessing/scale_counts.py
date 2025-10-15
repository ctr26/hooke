import numpy as np
import polars as pl
from loguru import logger
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
from vcb.preprocessing.utils import mean_relative_abs_error, right_skew, integer_only
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix


class TxDistributionHandler(BaseModel):
    """
    Transforms ground truth and predictions from source to log1p as needed; checks distributions while doing so
    """

    ground_truth: AnnotatedDataMatrix = Field(..., description="Ground truth data object")
    predictions: AnnotatedDataMatrix = Field(..., description="Predictions data object")
    desired_library_size: Optional[int] = Field(None, description="Desired library size for scaling")
    rescale_predictions: bool = Field(..., description="Whether to rescale predictions")
    log1p_transform_predictions: bool = Field(
        ..., description="Whether to apply log1p transform to predictions"
    )
    gt_before: Optional[Tuple[List, List]] = Field(
        None, description="Ground truth stats before transformation"
    )
    pred_before: Optional[Tuple[List, List]] = Field(
        None, description="Predictions stats before transformation"
    )
    gt_after: Optional[Tuple[List, List]] = Field(None, description="Ground truth stats after transformation")
    pred_after: Optional[Tuple[List, List]] = Field(
        None, description="Predictions stats after transformation"
    )

    def scale_both_as_needed(self):
        """
        Transforms gt and pred X, as specified such that they both end in log1p space; tracks and reports any unexpected distributional issues
        """
        # record distributions at the start
        self.gt_before = self.summarise_one(self.ground_truth.X)
        self.pred_before = self.summarise_one(self.predictions.X)

        scaler = RawCountScaler()

        # set library size
        if self.desired_library_size is not None:
            scaler.desired_library_size = self.desired_library_size
        else:
            scaler.fit(self.ground_truth.X)

        logger.info(
            f"Scaling as needed to library size: {scaler.desired_library_size}; rescale: {self.rescale_predictions}; will perform log1p of predictions: {self.log1p_transform_predictions}"
        )
        # ground truth should always be input as counts, and leave normalized and log transformed
        self.ground_truth.X = scaler.transform(self.ground_truth.X, is_log1p_transformed=False, rescale=True)

        self.predictions.X = scaler.transform(
            self.predictions.X,
            is_log1p_transformed=not self.log1p_transform_predictions,
            rescale=self.rescale_predictions,
        )

        # record distributions at end
        self.gt_after = self.summarise_one(self.ground_truth.X)
        self.pred_after = self.summarise_one(self.predictions.X)

        gt_summary = self.summarize_changes(self.gt_before, self.gt_after)
        pred_summary = self.summarize_changes(self.pred_before, self.pred_after)

        # sanity checks
        # we should never have zeros
        minimums = (
            gt_summary.join(pred_summary, on="Metric", suffix="_preds")
            .filter(pl.col("Metric") == "min")
            .drop("Metric")
        )
        if np.any(minimums < 0):
            logger.warning(
                f"Unexpected negative values detected, please check for distributional issues. {minimums}"
            )

        # for the ground truth, because we expect True count data, we will flag non-integers; or a lack of notable skew
        # for skew it's an imprecise science, given that datasets vary _a lot_ by sequencing type and specialization of cell type
        gt_in_has_only_integers = bool(gt_summary.filter(pl.col("Metric") == "integers")["Before"].item())
        gt_in_skew = gt_summary.filter(pl.col("Metric") == "mean/median")["Before"].item()

        if not gt_in_has_only_integers:
            logger.warning("Input ground truth might not be raw counts as it contains non-integer values")
        elif gt_in_skew < 1.1:
            logger.warning(
                f"Input ground truth might not be raw counts as it contains very little skew: {gt_in_skew}"
            )

        # for the predictions, there's two extra likely fail cases we can probably pick up by comparison to the ground truth
        # we've now log transformed twice (gt after is more similar to pred before than to pred after)
        vs_pred_before = mean_relative_abs_error(gt_summary["After"], pred_summary["Before"])
        vs_pred_after = mean_relative_abs_error(gt_summary["After"], pred_summary["After"])
        if vs_pred_before < vs_pred_after:
            logger.warning(
                "Predictions are more similar to normalized ground truth BEFORE normalization than after,"
                " please check for distributional issues; did we log transform twice?"
            )

        # we've not log transformed (pred after is more similar to gt before than gt after)
        vs_gt_before = mean_relative_abs_error(pred_summary["After"], gt_summary["Before"])
        vs_gt_after = mean_relative_abs_error(pred_summary["After"], gt_summary["After"])
        if vs_gt_before < vs_gt_after:
            logger.warning(
                "Final predictions are more similar to UNnormalized than normalized ground truth,"
                " please check for distributional issues; did we not log 1p transform?"
            )

        # final numbers
        logger.info(
            f"Data stats before and after transformations and normalization:\nGround Truth\n{gt_summary}\n\nPredictions\n{pred_summary}"
        )

    def summarise_one(self, data: np.ndarray) -> Tuple[List, List]:
        """stats on data and distribution;

        where skew is non-0 mean / median"""
        # CAUTION: updates here will require updates to scale_both_as_needed
        header = ["min", "max", "mean", "mean/median", "integers"]
        stats = [data.min(), data.max(), data.mean(), right_skew(data), integer_only(data)]
        return header, stats

    def summarize_changes(self, before_stats: List, after_stats: List) -> pl.DataFrame:
        """Summarize the data distribution."""
        header, before = before_stats
        _, after = after_stats

        return pl.DataFrame(
            {
                "Metric": header,
                "Before": before,
                "After": after,
            },
            schema={"Metric": pl.Utf8, "Before": pl.Float64, "After": pl.Float64},
        )


class RawCountScaler:
    """
    A scaler for Transcriptomics data.

    This scaler will rescale the data to a desired library size and log transform the data.
    """

    def __init__(self, desired_library_size: int | None = None):
        """If you know the desired library size, you can directly set it at initialization."""
        self._desired_library_size = None

    @property
    def desired_library_size(self) -> int | None:
        return self._desired_library_size

    @desired_library_size.setter
    def desired_library_size(self, value: int | None):
        assert value is not None and value > 0, (
            "The desired library size cannot be set to None and must be positive"
        )
        self._desired_library_size = value

    @property
    def fitted(self) -> bool:
        return self.desired_library_size is not None

    def fit(self, reference: np.ndarray):
        """If you don't know the desired library size, you can fit it from a reference array."""
        self.desired_library_size = np.median(np.sum(reference, axis=1))

    def transform(self, data: np.ndarray, is_log1p_transformed: bool = False, rescale: bool = True):
        if not self.fitted:
            raise ValueError(
                "The RawCountScaler is not fitted. "
                "Please pass a `reference` array or set `desired_library_size`"
            )

        d = data.copy()

        # Reset log1p transformation
        if is_log1p_transformed:
            d = np.exp(d) - 1

        if rescale:
            # Reset any prior library size scaling
            d = d / d.sum(axis=1, keepdims=True)

            # And then rescale and log transform the data
            d = d * self.desired_library_size

        # we'll always log transform for the final output
        d = np.log1p(d)

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
