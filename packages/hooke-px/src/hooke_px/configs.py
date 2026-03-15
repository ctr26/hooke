"""Hydra ConfigStore registration.

All configs defined in code, no YAML.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from hooke_px.schemas import EvalInput, InferenceInput


@dataclass
class InferenceConfig:
    """Inference job config."""

    checkpoint: str = "hooke-px/pretrain-checkpoint:latest"
    dataset: str = "/rxrx/data/pretraining"
    output_dir: str = "./outputs/inference"

    batch_size: int = 3
    num_workers: int = 100
    num_samples: int = 36

    partition: str = "hopper"
    gpus_per_node: int = 4

    def to_input(self) -> InferenceInput:
        """Convert to InferenceInput schema."""
        return InferenceInput(
            checkpoint_path=self.checkpoint,
            dataset_path=self.dataset,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_samples=self.num_samples,
            partition=self.partition,
            gpus_per_node=self.gpus_per_node,
        )


@dataclass
class EvalConfig:
    """Eval job config."""

    features_path: str = ""  # Set from inference output
    ground_truth_path: str = "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_2"
    split_path: str = "/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json"
    task_id: str = "virtual_map"
    split_index: int = 0

    def to_input(self) -> EvalInput:
        """Convert to EvalInput schema."""
        return EvalInput(
            features_path=self.features_path,
            ground_truth_path=self.ground_truth_path,
            split_path=self.split_path,
            task_id=self.task_id,
            split_index=self.split_index,
        )


@dataclass
class PipelineConfig:
    """Full pipeline config."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Pipeline mode
    run_eval: bool = False

    # Caching
    cache_dir: str = "/data/valence/cache"

    # Weave
    weave_project: str = "hooke-px"


def register_configs():
    """Register all configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="inference", node=InferenceConfig)
    cs.store(name="eval", node=EvalConfig)
    cs.store(name="pipeline", node=PipelineConfig)


# Register on import
register_configs()
