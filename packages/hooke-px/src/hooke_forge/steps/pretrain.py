"""Pretrain step — demo stub."""

import weave

from hooke_forge.schemas import ConditioningOutput, PretrainOutput


@weave.op()
def pretrain_step(input: ConditioningOutput, output_dir: str) -> PretrainOutput:
    return PretrainOutput(
        checkpoint_path=f"{output_dir}/checkpoint.pt",
        cell_types=input.cell_types,
        vocab_size=input.vocab_size,
        step=200000,
        test_compounds=input.test_compounds,
    )
