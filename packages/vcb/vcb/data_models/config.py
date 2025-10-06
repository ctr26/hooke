from typing import Annotated

import polars as pl
from pydantic import BaseModel, Field

from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite


class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation.
    """

    # NOTE (cwognum): Pydantic can't - to the best of my knowledge - automatically infer the subclass on deserialization.
    #  So we need to manually list all subclasses here and annotate the type with the discriminator.
    #  If this becomes a problem, we could look for inspiration in issues like these: https://github.com/pydantic/pydantic/issues/7366
    #  But it didn't think it was worth the overhead at this point.
    metric_suites: list[
        Annotated[RetrievalSuite | PerturbationEffectPredictionSuite, Field(discriminator="kind")]
    ]

    def execute(self) -> pl.DataFrame:
        results = []
        for suite in self.metric_suites:
            suite.prepare()
            result = suite.evaluate()
            results.append(result)

        return pl.concat(results, how="align")
