"""Evaluation schemas for the schema-governed pipeline."""

from pydantic import BaseModel


class EvalInput(BaseModel):
    features_path: str
    ground_truth_path: str
    split_path: str


class EvalOutput(BaseModel):
    metrics: dict[str, float]
