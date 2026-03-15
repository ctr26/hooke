"""Eval step — demo stub. Consumes from hooke-px via weave.ref()."""

import weave

from vcb.schemas import EvalInput, EvalOutput


@weave.op()
def eval_step(input: EvalInput) -> EvalOutput:
    return EvalOutput(
        metrics={"map_cosine": 0.85, "pearson": 0.72},
    )
