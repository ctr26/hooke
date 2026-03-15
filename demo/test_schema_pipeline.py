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
    def test_data_conf_defaults(self):
        d = DataConf()
        assert d.split_file == "data/splits/default.json"

    def test_conditioning_requires_data(self):
        with pytest.raises(ValidationError):
            ConditioningConf(split_path="/x", train_compounds=[], val_compounds=[], test_compounds=[])

    def test_pretrain_requires_conditioning(self):
        with pytest.raises(ValidationError):
            PretrainConf(cell_types=[], assay_types=[], vocab_size=2048, conditioning_path="/x")

    def test_finetuning_default_cell_type(self):
        pretrain = PretrainConf(
            conditioning=ConditioningConf(
                data=DataConf(), split_path="/x",
                train_compounds=[], val_compounds=[], test_compounds=[],
            ),
            cell_types=[], assay_types=[], vocab_size=2048, conditioning_path="/x",
        )
        f = FinetuningConf(pretrain=pretrain, checkpoint_path="/ckpt", step=100)
        assert f.target_cell_type == "ARPE19"

    def test_json_roundtrip(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        rebuilt = PretrainConf.model_validate_json(pretrain.model_dump_json())
        assert rebuilt == pretrain
        assert rebuilt.conditioning.data.split_file == pretrain.conditioning.data.split_file

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.project == "hooke-demo"


# -- Nested composition --


class TestComposition:
    def test_conditioning_nests_data(self):
        cond = splits_step(DataConf())
        assert isinstance(cond.data, DataConf)
        assert cond.data.split_file == "data/splits/default.json"

    def test_pretrain_nests_conditioning(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        assert isinstance(pretrain.conditioning, ConditioningConf)
        assert pretrain.conditioning.split_path == cond.split_path

    def test_finetuning_nests_pretrain(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        assert isinstance(finetune.pretrain, PretrainConf)
        assert finetune.pretrain.cell_types == pretrain.cell_types

    def test_full_lineage_traversal(self):
        cond = splits_step(DataConf())
        pretrain = conditioning_step(cond)
        finetune = pretrain_step(pretrain)
        inference = finetuning_step(finetune)
        ev = inference_step(inference)
        # Traverse all the way back to DataConf
        assert ev.inference.finetuning.pretrain.conditioning.data.split_file == "data/splits/default.json"


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
