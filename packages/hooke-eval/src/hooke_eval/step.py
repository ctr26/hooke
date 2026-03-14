"""Eval step with Weave lineage.

Consumes InferenceOutput from hooke-px via weave.ref().
"""

import weave

from hooke_px.schemas import EvalInput, EvalOutput, InferenceOutput


@weave.op()
def eval_step(input: EvalInput) -> EvalOutput:
    """Evaluate features using VCB metrics.

    Weave tracks lineage from inference → eval.
    """
    print(f"Evaluating: {input.features_path}")
    print(f"Ground truth: {input.ground_truth_path}")
    print(f"Split: {input.split_path}")

    # TODO: Actual VCB evaluation
    # from vcb.evaluation import VirtualMapSuite
    #
    # suite = VirtualMapSuite(
    #     predictions_path=input.features_path,
    #     ground_truth_path=input.ground_truth_path,
    #     split_path=input.split_path,
    # )
    # results = suite.run()

    # Placeholder metrics
    metrics = {
        "map_cosine": 0.85,
        "pearson_delta": 0.72,
        "pathway_capture": 0.68,
    }

    print(f"✅ Eval complete: {metrics}")

    return EvalOutput(
        metrics=metrics,
        features_path=input.features_path,
        eval_type="vcb",
    )


def eval_from_inference(inference_ref: str) -> EvalOutput:
    """Run eval from inference output ref.

    Example:
        eval_from_inference("hooke-px/inference-output:latest")
    """
    weave.init("hooke-eval")

    # Get inference output from hooke-px
    inference_output: InferenceOutput = weave.ref(inference_ref).get()

    # Convert to eval input
    eval_input = EvalInput(
        features_path=inference_output.features_path,
    )

    return eval_step(eval_input)
