"""Tests for accumulator pattern pipeline."""

import pytest

from schema_pipeline_accumulator import (
    PipelineState,
    conditioning_step,
    eval_step,
    finetuning_step,
    inference_step,
    pretrain_step,
    splits_step,
)


class TestAccumulation:
    def test_initial_state_is_mostly_none(self):
        state = PipelineState()
        assert state.split_path is None
        assert state.metrics is None

    def test_splits_sets_compounds(self):
        state = splits_step(PipelineState())
        assert state.train_compounds is not None
        assert state.cell_types is None  # not set yet

    def test_conditioning_preserves_splits(self):
        state = splits_step(PipelineState())
        state = conditioning_step(state)
        assert state.train_compounds is not None  # preserved
        assert state.cell_types is not None  # newly set

    def test_each_step_only_adds(self):
        state = PipelineState()
        filled = [len(state.model_dump(exclude_none=True))]
        for step in [splits_step, conditioning_step, pretrain_step, finetuning_step, inference_step, eval_step]:
            state = step(state)
            filled.append(len(state.model_dump(exclude_none=True)))
        # Each step should add fields, never remove
        assert filled == sorted(filled)

    def test_full_pipeline(self):
        state = PipelineState()
        for step in [splits_step, conditioning_step, pretrain_step, finetuning_step, inference_step, eval_step]:
            state = step(state)
        assert state.metrics is not None
        assert state.metrics["map_cosine"] == 0.85

    def test_json_roundtrip(self):
        state = PipelineState()
        for step in [splits_step, conditioning_step, pretrain_step]:
            state = step(state)
        rebuilt = PipelineState.model_validate_json(state.model_dump_json())
        assert rebuilt == state
