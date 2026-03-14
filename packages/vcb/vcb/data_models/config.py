import datetime
import subprocess
from pathlib import Path
from typing import Annotated, List, Union

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, computed_field, model_validator

from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.phenorescue import PhenorescueSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.metrics.suites.virtual_map import VirtualMapSuite
from vcb.data_models.split import Split
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.data_models.task.singles import SinglesTaskAdapter
from vcb.preprocessing.pipeline import PreprocessingPipeline, TranscriptomicsPreprocessingPipeline

# NOTE (cwognum): Pydantic can't - to the best of my knowledge - automatically infer the subclass on deserialization.
#  You could, and I have, use something like `BaseClass.__subclasses__()` to get the subclasses, but there is various
#  edge cases in which this fails. Importantly, you then need to explicitly import all subclasses to ensure they are defined.
#  If we're doing that anyways, we might as well list them here explicitly.
TASK_ADAPTERS_TYPE = Annotated[
    Union[DrugscreenTaskAdapter, SinglesTaskAdapter], Field(..., discriminator="kind")
]

METRIC_SUITES_TYPE = Annotated[
    Union[PerturbationEffectPredictionSuite, RetrievalSuite, PhenorescueSuite, VirtualMapSuite],
    Field(..., discriminator="kind"),
]

PREPROCESSING_PIPELINE_TYPE = Annotated[
    Union[TranscriptomicsPreprocessingPipeline, PreprocessingPipeline], Field(..., discriminator="kind")
]


class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation.
    """

    ground_truth: TASK_ADAPTERS_TYPE
    predictions: TASK_ADAPTERS_TYPE

    split_path: Path
    split_index: int
    use_validation_split: bool = False

    preprocessing_pipeline: PREPROCESSING_PIPELINE_TYPE | None = None

    copy_base_states_and_controls: bool = True

    metric_suites: List[METRIC_SUITES_TYPE]

    @model_validator(mode="after")
    def validate_consistency_between_tasks(self) -> "EvaluationConfig":
        """
        Assert the ground truth and predictions are the same type.
        """
        if not isinstance(self.ground_truth, type(self.predictions)):
            raise ValueError("Ground truth and predictions must be the same type")
        return self

    # For traceability sake, we'll also save some additional metadata
    @computed_field
    @property
    def timestamp(self) -> datetime.datetime:
        return datetime.datetime.now()

    @computed_field
    @property
    def git_commit(self) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    @computed_field
    @property
    def git_branch(self) -> str:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()

    @computed_field
    @property
    def git_status(self) -> str:
        return subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()

    def execute(self) -> pl.DataFrame:
        # Split the dataset
        split = Split.from_json(self.split_path)
        fold = split.folds[self.split_index]

        base_ctrl_indices = split.base_states + split.controls
        if self.use_validation_split:
            split_indices = base_ctrl_indices + fold.validation
        else:
            split_indices = base_ctrl_indices + fold.test

        self.ground_truth.dataset.filter(obs_indices=split_indices)

        gt = self.ground_truth.all_perturbed_obs["obs_id"].to_list()
        p = self.predictions.all_perturbed_obs["obs_id"].to_list()
        if len(gt) != len(p):
            raise RuntimeError(
                "Ground truth and predictions do not have the same number of observations. "
                f"Ground truth has {len(gt)} observations, predictions has {len(p)}."
            )
        if set(gt) != set(p):
            raise RuntimeError(
                "Ground truth and predictions do not have the same observations.\n"
                f"Ground truth - predictions = {set(gt) - set(p)}.\n"
                f"Predictions - ground truth = {set(p) - set(gt)}."
            )

        # Preprocess the ground truth and predictions
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline.transform(
                ground_truth=self.ground_truth.dataset,
                predictions=self.predictions.dataset,
            )

        if self.copy_base_states_and_controls:
            # Since the ground truth has already been filtered,
            # we need to find the index in the filtered dataset
            # for each of the indices in the original dataset.
            # Luckily, we know that all base states and control are contiguous and at the start.
            n_base_ctrl = len(base_ctrl_indices)

            gt_X_base_ctrl = self.ground_truth.dataset.X[:n_base_ctrl]
            gt_obs_base_ctrl = self.ground_truth.dataset.obs[:n_base_ctrl]

            p_X = self.predictions.dataset.X
            p_obs = self.predictions.dataset.obs

            intersection = set(p_obs.columns) & set(gt_obs_base_ctrl.columns)
            logger.warning(
                "Extending the predictions with the base states and controls from the ground truth. "
                f"For simplicity, we only keep the {len(intersection)} columns that overlap between the predictions and the ground truth! "
                f"Starting with {len(p_obs.columns)} columns for the predictions and {len(gt_obs_base_ctrl.columns)} columns for the ground truth. "
                "This should be ok when used within vcb, but if you run into issues, please double check this part."
            )

            # Extend the predictions with the base states and controls
            self.predictions.dataset.update(
                obs=pl.concat(
                    [p_obs.select(intersection), gt_obs_base_ctrl.select(intersection)],
                    how="vertical_relaxed",
                ),
                X=np.concatenate([p_X, gt_X_base_ctrl], axis=0),
            )
        # Sort the ground truth and predictions by their obs id.
        # For most tasks, this shouldn't matter.
        self.ground_truth.dataset.filter(
            obs_indices=self.ground_truth.dataset.obs["obs_id"].arg_sort().to_list()
        )
        self.predictions.dataset.filter(
            obs_indices=self.predictions.dataset.obs["obs_id"].arg_sort().to_list()
        )

        # final task adapter specific preparation
        self.ground_truth.prepare()
        self.predictions.prepare()

        # Evaluate the predictions
        results = []
        for suite in self.metric_suites:
            result = suite.evaluate(ground_truth=self.ground_truth, predictions=self.predictions)
            results.append(result)

        return pl.concat(results, how="align")
