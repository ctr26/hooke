from typing import Annotated, List, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.log1p import InverseLog1pStep, Log1pStep
from vcb.preprocessing.steps.match_genes import MatchGenesStep
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep
from vcb.preprocessing.utils import (
    distribution_summary_similarity,
    integer_only,
    right_skew,
    summaries_to_table,
    summarize_distribution,
)


class PreprocessingPipeline(BaseModel):
    """
    A preprocessing pipeline.
    """

    kind: Literal["base"] = "base"

    # NOTE (cwognum): Pydantic can't - to the best of my knowledge - automatically infer the subclass on deserialization.
    #  You could, and I have, use something like `BaseClass.__subclasses__()` to get the subclasses, but there is various
    #  edge cases in which this fails. Importantly, you then need to explicitly import all subclasses to ensure they are defined.
    #  If we're doing that anyways, we might as well list them here explicitly.
    steps: List[
        Annotated[
            Union[ScaleCountsStep, MatchGenesStep, Log1pStep, InverseLog1pStep],
            Field(..., discriminator="kind"),
        ]
    ]

    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        for step in self.steps:
            step.fit(ground_truth, predictions)
            step.transform(ground_truth, predictions)


class TranscriptomicsPreprocessingPipeline(PreprocessingPipeline):
    """
    A preprocessing pipeline for transcriptomics data.

    In addition to the base preprocessing pipeline, this pipeline will also summarize the distribution of the data
    and run some sanity checks on the data.
    """

    kind: Literal["transcriptomics"] = "transcriptomics"

    def transform(self, ground_truth: AnnotatedDataMatrix, predictions: AnnotatedDataMatrix):
        # for the ground truth, because we expect raw count data, we will flag non-integers; or a lack of notable skew
        # for skew it's an imprecise science, given that datasets vary _a lot_ by sequencing type and specialization of cell type
        if not integer_only(ground_truth.X):
            logger.warning("Input ground truth might not be raw counts as it contains non-integer values")
        if right_skew(ground_truth.X) < 1.1:
            logger.warning("Input ground truth might not be raw counts as it contains very little skew")

        # We don't expect negative values
        if ground_truth.X.min() < 0 or predictions.X.min() < 0:
            logger.warning(
                "Unexpected negative values detected prior to transformations, please check for distributional issues."
            )

        # Summarize the distribution of the data prior to any transformations
        gt_before = summarize_distribution(ground_truth.X)
        pr_before = summarize_distribution(predictions.X)

        # Apply the transformations
        super().transform(ground_truth, predictions)

        # We still don't expect negative values
        if ground_truth.X.min() < 0 or predictions.X.min() < 0:
            logger.warning(
                "Unexpected negative values detected after transformations, please check for distributional issues."
            )

        gt_after = summarize_distribution(ground_truth.X)
        pr_after = summarize_distribution(predictions.X)

        # for the predictions, there's two extra likely fail cases we can probably pick up by comparison to the ground truth
        # we've now log transformed twice (gt after is more similar to pred before than to pred after)
        vs_pred_before = distribution_summary_similarity(gt_after, pr_before)
        vs_pred_after = distribution_summary_similarity(gt_after, pr_after)

        if vs_pred_before < vs_pred_after:
            logger.warning(
                "Predictions are more similar to ground truth BEFORE normalization than after, "
                "please check for distributional issues; did we log transform twice?"
            )

        # we've not log transformed (pred after is more similar to gt before than gt after)
        vs_gt_before = distribution_summary_similarity(pr_after, gt_before)
        vs_gt_after = distribution_summary_similarity(pr_after, gt_after)
        if vs_gt_before < vs_gt_after:
            logger.warning(
                "Final predictions are more similar to UNnormalized than normalized ground truth,"
                " please check for distributional issues; did we not log1p transform?"
            )

        logger.info(
            f"Data stats before and after transformations and normalization:\n"
            f"Ground Truth\n{summaries_to_table({'Before': gt_before, 'After': gt_after})}\n\n"
            f"Predictions\n{summaries_to_table({'Before': pr_before, 'After': pr_after})}"
        )
