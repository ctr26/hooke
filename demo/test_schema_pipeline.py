"""Tests for schema pipeline demo."""

import pytest
from pydantic import ValidationError

from schema_pipeline import (
    ConditioningConf,
    EvalConf,
    ResultsConf,
    InferenceConf,
    PipelineConfig,
    PretrainConf,
    SplitsConf,
    conditioning_step,
    eval_step,
    inference_step,
    pretrain_step,
    splits_step,
)


# -- Schema validation --


class TestSchemaValidation:
    def test_conditioning_input_missing_field(self):
        with pytest.raises(ValidationError):
            ConditioningConf(split_path="/x")

    def test_pretrain_input_wrong_type(self):
        with pytest.raises(ValidationError):
            PretrainConf(
                split_path="/x", train_compounds=[], val_compounds=[],
                test_compounds=[], cell_types=[], assay_types=[],
                vocab_size="bad", conditioning_path="/x",
            )

    def test_splits_input_defaults(self):
        s = SplitsConf()
        assert s.split_file == "data/splits/default.json"

    def test_json_roundtrip(self):
        o = InferenceConf(
            checkpoint_path="/ckpt", cell_types=["A"], vocab_size=1024,
            step=100, test_compounds=["c"],
        )
        assert InferenceConf.model_validate_json(o.model_dump_json()) == o

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.output_dir == "outputs/demo"
        assert cfg.project == "hooke-demo"

    def test_pipeline_config_override(self):
        cfg = PipelineConfig(output_dir="/tmp/run", project="test")
        assert cfg.output_dir == "/tmp/run"


# -- Schema chaining: step output = next step's input --


class TestSchemaChaining:
    def test_splits_returns_conditioning_input(self):
        result = splits_step(SplitsConf())
        assert isinstance(result, ConditioningConf)
        assert result.split_path == "data/splits/default.json"

    def test_conditioning_returns_pretrain_input(self):
        cond_in = splits_step(SplitsConf())
        pretrain_in = conditioning_step(cond_in)
        assert isinstance(pretrain_in, PretrainConf)
        assert pretrain_in.split_path == cond_in.split_path
        assert pretrain_in.test_compounds == cond_in.test_compounds

    def test_pretrain_returns_inference_input(self):
        cond_in = splits_step(SplitsConf())
        pretrain_in = conditioning_step(cond_in)
        inference_in = pretrain_step(pretrain_in)
        assert isinstance(inference_in, InferenceConf)
        assert inference_in.cell_types == pretrain_in.cell_types

    def test_inference_returns_eval_input(self):
        cond_in = splits_step(SplitsConf())
        pretrain_in = conditioning_step(cond_in)
        inference_in = pretrain_step(pretrain_in)
        eval_in = inference_step(inference_in)
        assert isinstance(eval_in, EvalConf)
        assert eval_in.num_samples == len(inference_in.test_compounds) * 100


# -- End to end --


class TestPipeline:
    def test_full_chain(self):
        cond_in = splits_step(SplitsConf())
        pretrain_in = conditioning_step(cond_in)
        inference_in = pretrain_step(pretrain_in)
        eval_in = inference_step(inference_in)
        result = eval_step(eval_in)
        assert isinstance(result, ResultsConf)
        assert "map_cosine" in result.metrics
