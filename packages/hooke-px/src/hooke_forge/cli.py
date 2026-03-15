"""CLI entry point — hydra-zen + Pydantic schemas."""

import hydra
from hydra_zen import builds, store, zen

from hooke_forge.schemas import InferenceInput


# Register Pydantic schemas with Hydra ConfigStore
InferenceConf = builds(InferenceInput, populate_full_signature=True)
store(InferenceConf, name="inference", group="schema")


def run_inference(cfg: InferenceInput) -> None:
    from hooke_forge.steps.inference import inference_step

    result = inference_step(cfg)
    print(f"Output: {result}")


def run_pipeline(output_dir: str = "outputs/demo") -> None:
    from hooke_forge.orchestrator import run_pipeline

    run_pipeline(output_dir)


# Hydra-zen task functions
inference_task = zen(run_inference)
pipeline_task = zen(run_pipeline)

store(builds(run_inference, populate_full_signature=True), name="inference")
store(builds(run_pipeline, populate_full_signature=True), name="pipeline")
store.add_to_hydra_store()


@hydra.main(config_path=None, config_name="inference", version_base=None)
def main(cfg):
    run_inference(InferenceInput(**cfg))


if __name__ == "__main__":
    main()
