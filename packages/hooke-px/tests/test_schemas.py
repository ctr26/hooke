"""Tests for pipeline schemas — validation happy + error paths."""

import pytest
from pydantic import ValidationError

from hooke_forge.schemas import (
    ConditioningOutput,
    InferenceInput,
    InferenceOutput,
    PretrainOutput,
    SplitsOutput,
)


class TestSplitsOutput:
    def test_valid(self):
        s = SplitsOutput(
            split_path="/data/split.json",
            train_compounds=["cpd_001"],
            val_compounds=["cpd_002"],
            test_compounds=["cpd_003"],
        )
        assert s.split_path == "/data/split.json"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            SplitsOutput(split_path="/data/split.json")

    def test_empty_compounds(self):
        s = SplitsOutput(
            split_path="/data/split.json",
            train_compounds=[],
            val_compounds=[],
            test_compounds=[],
        )
        assert s.train_compounds == []


class TestConditioningOutput:
    def test_valid(self):
        c = ConditioningOutput(
            split_path="/data/split.json",
            train_compounds=["cpd_001"],
            val_compounds=[],
            test_compounds=[],
            cell_types=["ARPE19"],
            assay_types=["cell_paint"],
            vocab_size=2048,
            conditioning_path="/data/cond.json",
        )
        assert c.vocab_size == 2048

    def test_wrong_type_vocab_size(self):
        with pytest.raises(ValidationError):
            ConditioningOutput(
                split_path="/data/split.json",
                train_compounds=[],
                val_compounds=[],
                test_compounds=[],
                cell_types=[],
                assay_types=[],
                vocab_size="not_an_int",
                conditioning_path="/data/cond.json",
            )


class TestPretrainOutput:
    def test_valid(self):
        p = PretrainOutput(
            checkpoint_path="/ckpt.pt",
            cell_types=["ARPE19"],
            vocab_size=2048,
            step=100000,
            test_compounds=["cpd_001"],
        )
        assert p.step == 100000


class TestInferenceInput:
    def test_defaults(self):
        i = InferenceInput(
            checkpoint_path="/ckpt.pt",
            dataset_path="/data",
        )
        assert i.batch_size == 32
        assert i.num_workers == 4

    def test_override_defaults(self):
        i = InferenceInput(
            checkpoint_path="/ckpt.pt",
            dataset_path="/data",
            batch_size=64,
            num_workers=8,
        )
        assert i.batch_size == 64


class TestInferenceOutput:
    def test_valid(self):
        o = InferenceOutput(features_path="/features.npy", num_samples=1000)
        assert o.num_samples == 1000


class TestSchemaChaining:
    """Test that step N output can construct step N+1 input."""

    def test_splits_to_conditioning(self):
        splits = SplitsOutput(
            split_path="/split.json",
            train_compounds=["a"],
            val_compounds=["b"],
            test_compounds=["c"],
        )
        cond = ConditioningOutput(
            **splits.model_dump(),
            cell_types=["ARPE19"],
            assay_types=["cell_paint"],
            vocab_size=2048,
            conditioning_path="/cond.json",
        )
        assert cond.split_path == splits.split_path

    def test_conditioning_to_pretrain(self):
        cond = ConditioningOutput(
            split_path="/split.json",
            train_compounds=["a"],
            val_compounds=["b"],
            test_compounds=["c"],
            cell_types=["ARPE19"],
            assay_types=["cell_paint"],
            vocab_size=2048,
            conditioning_path="/cond.json",
        )
        pretrain = PretrainOutput(
            checkpoint_path="/ckpt.pt",
            cell_types=cond.cell_types,
            vocab_size=cond.vocab_size,
            step=200000,
            test_compounds=cond.test_compounds,
        )
        assert pretrain.cell_types == cond.cell_types

    def test_roundtrip_serialization(self):
        original = InferenceInput(checkpoint_path="/ckpt.pt", dataset_path="/data")
        rebuilt = InferenceInput.model_validate_json(original.model_dump_json())
        assert rebuilt == original
