"""Pipeline orchestrator — demo stub.

Chains all steps with Weave lineage across hooke-px and hooke-eval.
"""

import weave

from hooke_forge.schemas import InferenceInput
from hooke_forge.steps.splits import splits_step
from hooke_forge.steps.conditioning import conditioning_step
from hooke_forge.steps.pretrain import pretrain_step
from hooke_forge.steps.inference import inference_step
from vcb.step import eval_step
from vcb.schemas import EvalInput


def run_pipeline(output_dir: str = "outputs/demo") -> None:
    weave.init("hooke-px")

    splits = splits_step("data/splits/default.json", f"{output_dir}/splits")
    config = conditioning_step(splits, f"{output_dir}/cond")
    checkpoint = pretrain_step(config, f"{output_dir}/pretrain")

    inference_out = inference_step(
        InferenceInput(
            checkpoint_path=checkpoint.checkpoint_path,
            dataset_path=f"{output_dir}/data",
        )
    )

    # Cross-project: publish for hooke-eval
    weave.publish(inference_out, name="inference-output")

    # Eval step (would run in hooke-eval project)
    eval_result = eval_step(
        EvalInput(
            features_path=inference_out.features_path,
            ground_truth_path=f"{output_dir}/ground_truth",
            split_path=splits.split_path,
        )
    )

    print(f"Pipeline complete: {eval_result.metrics}")


if __name__ == "__main__":
    run_pipeline()
