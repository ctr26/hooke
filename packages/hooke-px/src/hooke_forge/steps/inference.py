"""Inference step — demo stub with Submitit + joblib placeholders."""

import weave

from hooke_forge.schemas import InferenceInput, InferenceOutput


@weave.op()
def inference_step(input: InferenceInput) -> InferenceOutput:
    # TODO: submitit GPU job submission
    # TODO: joblib.Memory cache at /data/valence/cache
    return InferenceOutput(
        features_path=f"{input.dataset_path}/features.npy",
        num_samples=1000,
    )
