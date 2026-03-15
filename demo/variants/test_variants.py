"""Tests for all pipeline variants."""

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
        assert "map_cosine" in result.metrics

    def test_outputs_are_data_not_paths(self):
        cond = nested.splits_step(DataConf())
        pretrain = nested.conditioning_step(cond)
        assert isinstance(pretrain.conditioning_weights, list)
        assert all(isinstance(w, float) for w in pretrain.conditioning_weights)

    def test_features_are_embeddings(self):
        cond = nested.splits_step(DataConf())
        pretrain = nested.conditioning_step(cond)
        finetune = nested.pretrain_step(pretrain)
        inference = nested.finetuning_step(finetune)
        ev = nested.inference_step(inference)
        assert len(ev.features) == ev.num_samples
        assert len(ev.features[0]) == 3  # 3-dim embeddings

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
        assert "map_cosine" in (state.metrics or {})

    def test_outputs_are_data(self):
        state = PipelineState()
        for step in accumulator.STEPS[:3]:
            state = step(state)
        assert isinstance(state.model_weights, list)
        assert all(isinstance(w, float) for w in state.model_weights)

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
        assert "map_cosine" in (result.metrics or {})

    def test_partial_pipe(self):
        result = functional_pipe.pipe(functional_pipe.PIPELINE[:3], PipelineState())
        assert result.model_weights is not None
        assert result.finetuned_weights is None


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
        assert "map_cosine" in (result.metrics or {})

    def test_partial_chain(self):
        result = method_chain.Pipeline().split().condition().pretrain()
        assert result.model_weights is not None
        assert result.metrics is None


class TestWeaveRefs:
    def test_full_chain(self):
        cond = weave_refs.splits_step(DataConf())
        pretrain = weave_refs.conditioning_step(cond)
        finetune = weave_refs.pretrain_step(pretrain)
        inference = weave_refs.finetuning_step(finetune)
        ev = weave_refs.inference_step(inference)
        result = weave_refs.eval_step(ev)
        assert "map_cosine" in result.metrics

    def test_outputs_are_data(self):
        cond = weave_refs.splits_step(DataConf())
        pretrain = weave_refs.conditioning_step(cond)
        finetune = weave_refs.pretrain_step(pretrain)
        assert isinstance(finetune.model_weights, list)


class TestAllVariantsAgree:
    def test_same_metrics_keys(self):
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

        # All produce the same metric keys
        assert set(nested_result.metrics.keys()) == set((state.metrics or {}).keys())
        assert set(nested_result.metrics.keys()) == set((pipe_result.metrics or {}).keys())
        assert set(nested_result.metrics.keys()) == set((chain_result.metrics or {}).keys())
