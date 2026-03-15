"""Tests for eval schemas."""

import pytest
from pydantic import ValidationError

from vcb.schemas import EvalInput, EvalOutput


class TestEvalInput:
    def test_valid(self):
        e = EvalInput(
            features_path="/features.npy",
            ground_truth_path="/gt.npy",
            split_path="/split.json",
        )
        assert e.features_path == "/features.npy"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            EvalInput(features_path="/features.npy")


class TestEvalOutput:
    def test_valid(self):
        o = EvalOutput(metrics={"map_cosine": 0.85, "pearson": 0.72})
        assert o.metrics["map_cosine"] == 0.85

    def test_empty_metrics(self):
        o = EvalOutput(metrics={})
        assert o.metrics == {}

    def test_serialization_roundtrip(self):
        o = EvalOutput(metrics={"acc": 0.95})
        rebuilt = EvalOutput.model_validate_json(o.model_dump_json())
        assert rebuilt == o
