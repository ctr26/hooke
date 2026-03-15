"""Conditioning step — demo stub."""

import weave

from hooke_forge.schemas import ConditioningOutput, SplitsOutput


@weave.op()
def conditioning_step(input: SplitsOutput, output_dir: str) -> ConditioningOutput:
    return ConditioningOutput(
        split_path=input.split_path,
        train_compounds=input.train_compounds,
        val_compounds=input.val_compounds,
        test_compounds=input.test_compounds,
        cell_types=["ARPE19", "HUVEC"],
        assay_types=["cell_paint", "brightfield"],
        vocab_size=2048,
        conditioning_path=f"{output_dir}/conditioning.json",
    )
