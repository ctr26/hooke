import datetime
import subprocess
from pathlib import Path
from typing import Annotated, List, Union

import polars as pl
from pydantic import BaseModel, Field, computed_field, model_validator

from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.split import Split
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.preprocessing.pipeline import PreprocessingPipeline, TranscriptomicsPreprocessingPipeline
from vcb.data_models.task.singles import UnseenSingleTaskAdapter

# NOTE (cwognum): Pydantic can't - to the best of my knowledge - automatically infer the subclass on deserialization.
#  You could, and I have, use something like `BaseClass.__subclasses__()` to get the subclasses, but there is various
#  edge cases in which this fails. Importantly, you then need to explicitly import all subclasses to ensure they are defined.
#  If we're doing that anyways, we might as well list them here explicitly.

TASK_ADAPTERS_TYPE = Annotated[
    Union[DrugscreenTaskAdapter, UnseenSingleTaskAdapter], Field(..., discriminator="kind")
]

METRIC_SUITES_TYPE = Annotated[
    Union[PerturbationEffectPredictionSuite, RetrievalSuite], Field(..., discriminator="kind")
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

        if self.use_validation_split:
            split_indices = fold.validation + split.base_states
        else:
            split_indices = fold.test + split.base_states

        self.ground_truth.dataset.filter(obs_indices=split_indices)

        gt = self.ground_truth.get_all_perturbed_obs()["obs_id"].to_list()
        p = self.predictions.get_all_perturbed_obs()["obs_id"].to_list()
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

        # final task adapter specific preparation
        self.ground_truth.prepare()
        self.predictions.prepare()

        # Evaluate the predictions
        results = []
        for suite in self.metric_suites:
            result = suite.evaluate(ground_truth=self.ground_truth, predictions=self.predictions)
            results.append(result)

        return pl.concat(results, how="align")
