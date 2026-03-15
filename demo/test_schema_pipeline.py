"""Tests for schema pipeline demo."""

import pytest
from pydantic import ValidationError

from schema_pipeline import (
    ConditioningOutput,
    EvalOutput,
    InferenceOutput,
    PretrainOutput,
    SplitsOutput,
    conditioning_step,
    eval_step,
    inference_step,
    pretrain_step,
    splits_step,
)


# -- Schema validation --


class TestSchemaValidation:
    def test_splits_missing_field(self):
        with pytest.raises(ValidationError):
            SplitsOutput(split_path="/x")

    def test_conditioning_wrong_type(self):
        with pytest.raises(ValidationError):
            ConditioningOutput(
                split_path="/x", train_compounds=[], val_compounds=[],
                test_compounds=[], cell_types=[], assay_types=[],
                vocab_size="bad", conditioning_path="/x",
            )

    def test_defaults_not_required(self):
        s = SplitsOutput(
            split_path="/x", train_compounds=[], val_compounds=[], test_compounds=[],
        )
        assert s.split_path == "/x"

    def test_json_roundtrip(self):
        o = PretrainOutput(
            checkpoint_path="/ckpt", cell_types=["A"], vocab_size=1024,
            step=100, test_compounds=["c"],
        )
        assert PretrainOutput.model_validate_json(o.model_dump_json()) == o


# -- Schema chaining --


class TestSchemaChaining:
    def test_splits_fields_carry_to_conditioning(self):
        splits = splits_step()
        cond = conditioning_step(splits)
        assert cond.split_path == splits.split_path
        assert cond.test_compounds == splits.test_compounds

    def test_conditioning_fields_carry_to_pretrain(self):
        cond = conditioning_step(splits_step())
        pretrain = pretrain_step(cond)
        assert pretrain.cell_types == cond.cell_types
        assert pretrain.vocab_size == cond.vocab_size

    def test_pretrain_compounds_carry_to_inference(self):
        pretrain = pretrain_step(conditioning_step(splits_step()))
        inference = inference_step(pretrain)
        assert inference.num_samples == len(pretrain.test_compounds) * 100


# -- End to end --


class TestPipeline:
    def test_full_chain(self):
        splits = splits_step()
        cond = conditioning_step(splits)
        pretrain = pretrain_step(cond)
        inference = inference_step(pretrain)
        result = eval_step(inference.features_path, splits.split_path)
        assert isinstance(result, EvalOutput)
        assert "map_cosine" in result.metrics
