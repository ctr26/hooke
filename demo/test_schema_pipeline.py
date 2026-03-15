"""Tests for schema pipeline demo."""

import pytest
from pydantic import ValidationError

from schema_pipeline import (
    ConditioningConf,
    DataConf,
    EvalConf,
    FinetuningConf,
    InferenceConf,
    PipelineConfig,
    PretrainConf,
    ResultsConf,
    conditioning_step,
    eval_step,
    finetuning_step,
    inference_step,
    pretrain_step,
    splits_step,
)


# -- Schema validation --


class TestSchemaValidation:
    def test_conditioning_missing_field(self):
        with pytest.raises(ValidationError):
            ConditioningConf(split_path="/x")

    def test_pretrain_wrong_type(self):
        with pytest.raises(ValidationError):
            PretrainConf(
                split_path="/x", train_compounds=[], val_compounds=[],
                test_compounds=[], cell_types=[], assay_types=[],
                vocab_size="bad", conditioning_path="/x",
            )

    def test_data_conf_defaults(self):
        d = DataConf()
        assert d.split_file == "data/splits/default.json"

    def test_finetuning_conf_default_cell_type(self):
        f = FinetuningConf(
            checkpoint_path="/ckpt", cell_types=["A"], vocab_size=1024,
            step=100, test_compounds=["c"],
        )
        assert f.target_cell_type == "ARPE19"

    def test_json_roundtrip(self):
        o = InferenceConf(
            checkpoint_path="/ckpt", cell_types=["A"], vocab_size=1024,
            step=100, test_compounds=["c"], target_cell_type="HUVEC",
        )
        assert InferenceConf.model_validate_json(o.model_dump_json()) == o

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.output_dir == "outputs/demo"
        assert cfg.project == "hooke-demo"


# -- Schema chaining --


class TestSchemaChaining:
    def test_splits_returns_conditioning_conf(self):
        result = splits_step(DataConf())
        assert isinstance(result, ConditioningConf)

    def test_conditioning_returns_pretrain_conf(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        assert isinstance(pretrain, PretrainConf)
        assert pretrain.split_path == cond.split_path

    def test_pretrain_returns_finetuning_conf(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        assert isinstance(finetune, FinetuningConf)
        assert finetune.cell_types == pretrain.cell_types

    def test_finetuning_returns_inference_conf(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        inference = finetuning_step(finetune)
        assert isinstance(inference, InferenceConf)
        assert inference.target_cell_type == finetune.target_cell_type

    def test_inference_returns_eval_conf(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        inference = finetuning_step(finetune)
        ev = inference_step(inference)
        assert isinstance(ev, EvalConf)


# -- End to end --


class TestPipeline:
    def test_full_chain(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        inference = finetuning_step(finetune)
        ev = inference_step(inference)
        result = eval_step(ev)
        assert isinstance(result, ResultsConf)
        assert "map_cosine" in result.metrics
