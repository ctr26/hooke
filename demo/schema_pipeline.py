#!/usr/bin/env python
"""Demo: Schema-governed pipeline — each step returns the next step's input.

No external services needed. Pure Pydantic validation + schema chaining.

    python demo/schema_pipeline.py
"""

from hooke_forge.schemas import (
    ConditioningOutput,
    EvalInput,
    InferenceInput,
    InferenceOutput,
    PretrainOutput,
    SplitsOutput,
)
from vcb.schemas import EvalInput as VCBEvalInput, EvalOutput


# -- Steps: each returns next step's input schema --


def splits_step() -> SplitsOutput:
    print("[1/5] splits_step → SplitsOutput")
    return SplitsOutput(
        split_path="/data/splits/v1.json",
        train_compounds=["cpd_001", "cpd_002", "cpd_003"],
        val_compounds=["cpd_004"],
        test_compounds=["cpd_005", "cpd_006"],
    )


def conditioning_step(input: SplitsOutput) -> ConditioningOutput:
    print("[2/5] conditioning_step → ConditioningOutput")
    return ConditioningOutput(
        **input.model_dump(),
        cell_types=["ARPE19", "HUVEC", "HepG2"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path="/data/conditioning/v1.json",
    )


def pretrain_step(input: ConditioningOutput) -> PretrainOutput:
    print("[3/5] pretrain_step → PretrainOutput")
    return PretrainOutput(
        checkpoint_path="/checkpoints/pretrain_200k.pt",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200_000,
        test_compounds=input.test_compounds,
    )


def inference_step(input: PretrainOutput) -> InferenceOutput:
    print("[4/5] inference_step → InferenceOutput")
    return InferenceOutput(
        features_path="/outputs/features.npy",
        num_samples=len(input.test_compounds) * 100,
    )


def eval_step(input: VCBEvalInput) -> EvalOutput:
    print("[5/5] eval_step → EvalOutput")
    return EvalOutput(metrics={"map_cosine": 0.85, "pearson": 0.72})


# -- Pipeline --


def main():
    print("=" * 50)
    print("Schema-governed pipeline demo")
    print("=" * 50)
    print()

    splits = splits_step()
    cond = conditioning_step(splits)
    pretrain = pretrain_step(cond)
    inference = inference_step(pretrain)

    result = eval_step(
        VCBEvalInput(
            features_path=inference.features_path,
            ground_truth_path="/data/ground_truth.npy",
            split_path=splits.split_path,
        )
    )

    print()
    print(f"Result: {result.metrics}")
    print()

    # Show schema chaining works
    print("Schema chain proof:")
    print(f"  SplitsOutput fields carry to ConditioningOutput: {cond.split_path == splits.split_path}")
    print(f"  ConditioningOutput.cell_types carry to PretrainOutput: {pretrain.cell_types == cond.cell_types}")
    print(f"  InferenceOutput.features_path feeds EvalInput: {result.metrics['map_cosine']}")
    print()

    # Roundtrip serialization
    json_str = pretrain.model_dump_json(indent=2)
    rebuilt = PretrainOutput.model_validate_json(json_str)
    print(f"JSON roundtrip: {rebuilt == pretrain}")


if __name__ == "__main__":
    main()
