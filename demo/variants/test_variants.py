"""Tests for all pipeline variants."""

import pytest

from variants.schemas import DataConf, PipelineState
from variants import nested, accumulator, functional_pipe, method_chain, weave_refs


class TestNested:
    def test_full_chain(self):
        cond = nested.splits_step(DataConf())
        pretrain = nested.conditioning_step(cond)
        finetune = nested.pretrain_step(pretrain)
        inference = nested.finetuning_step(finetune)
        ev = nested.inference_step(inference)
        result = nested.eval_step(ev)
        assert result.metrics["map_cosine"] == 0.85

    def test_lineage_traversal(self):
        cond = nested.splits_step(DataConf())
        pretrain = nested.conditioning_step(cond)
        finetune = nested.pretrain_step(pretrain)
        inference = nested.finetuning_step(finetune)
        ev = nested.inference_step(inference)
        assert ev.inference.finetuning.pretrain.conditioning.data.split_file == "data/splits/default.json"


class TestAccumulator:
    def test_full_loop(self):
        state = PipelineState()
        for step in accumulator.STEPS:
            state = step(state)
        assert state.metrics["map_cosine"] == 0.85

    def test_monotonic_growth(self):
        state = PipelineState()
        counts = [len(state.model_dump(exclude_none=True))]
        for step in accumulator.STEPS:
            state = step(state)
            counts.append(len(state.model_dump(exclude_none=True)))
        assert counts == sorted(counts)


class TestFunctionalPipe:
    def test_pipe(self):
        result = functional_pipe.pipe(functional_pipe.PIPELINE, PipelineState())
        assert result.metrics["map_cosine"] == 0.85

    def test_partial_pipe(self):
        result = functional_pipe.pipe(functional_pipe.PIPELINE[:3], PipelineState())
        assert result.pretrain_checkpoint is not None
        assert result.finetune_checkpoint is None


class TestMethodChain:
    def test_full_chain(self):
        result = (
            method_chain.Pipeline()
            .split()
            .condition()
            .pretrain()
            .finetune()
            .infer()
            .evaluate()
        )
        assert result.metrics["map_cosine"] == 0.85

    def test_partial_chain(self):
        result = method_chain.Pipeline().split().condition().pretrain()
        assert result.pretrain_checkpoint is not None
        assert result.metrics is None


class TestWeaveRefs:
    def test_full_chain(self):
        cond = weave_refs.splits_step(DataConf())
        pretrain = weave_refs.conditioning_step(cond)
        finetune = weave_refs.pretrain_step(pretrain)
        inference = weave_refs.finetuning_step(finetune)
        ev = weave_refs.inference_step(inference)
        result = weave_refs.eval_step(ev)
        assert result.metrics["map_cosine"] == 0.85

    def test_lineage_traversal(self):
        cond = weave_refs.splits_step(DataConf())
        pretrain = weave_refs.conditioning_step(cond)
        finetune = weave_refs.pretrain_step(pretrain)
        inference = weave_refs.finetuning_step(finetune)
        ev = weave_refs.inference_step(inference)
        assert ev.inference.finetuning.pretrain.conditioning.data.split_file == "data/splits/default.json"


class TestAllVariantsAgree:
    def test_same_final_metrics(self):
        # Nested
        cond = nested.splits_step(DataConf())
        pretrain = nested.conditioning_step(cond)
        finetune = nested.pretrain_step(pretrain)
        inference = nested.finetuning_step(finetune)
        ev = nested.inference_step(inference)
        nested_result = nested.eval_step(ev)

        # Accumulator
        state = PipelineState()
        for step in accumulator.STEPS:
            state = step(state)

        # Pipe
        pipe_result = functional_pipe.pipe(functional_pipe.PIPELINE, PipelineState())

        # Method chain
        chain_result = method_chain.Pipeline().split().condition().pretrain().finetune().infer().evaluate()

        assert nested_result.metrics == state.metrics == pipe_result.metrics == chain_result.metrics
