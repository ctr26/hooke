"""Splits step — demo stub."""

import weave

from hooke_forge.schemas import SplitsOutput


@weave.op()
def splits_step(split_file: str, output_dir: str) -> SplitsOutput:
    return SplitsOutput(
        split_path=f"{output_dir}/split.json",
        train_compounds=["cpd_001", "cpd_002"],
        val_compounds=["cpd_003"],
        test_compounds=["cpd_004", "cpd_005"],
    )
