"""Pydantic schemas for hooke-px pipeline.

Each step's output = next step's input.
"""

from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """Input for inference step."""

    checkpoint_path: str = Field(..., description="W&B artifact ref or local path")
    dataset_path: str = Field(..., description="Path to dataset (parquet)")
    output_dir: str = Field(..., description="Output directory for features")

    # Job config
    batch_size: int = Field(default=3, description="Batch size per GPU")
    num_workers: int = Field(default=100, description="Number of parallel workers")
    num_samples: int = Field(default=36, description="Samples per well")

    # Representations
    representations: list[str] | None = Field(default=None, description="Representations to extract (auto-detected if None)")
    tx_zarr_path: str = Field(default="", description="Path to tx feature zarr (required for tx modality)")

    # SLURM config
    partition: str = Field(default="hopper", description="SLURM partition")
    gpus_per_node: int = Field(default=4, description="GPUs per node")
    qos: str | None = Field(default=None, description="SLURM QOS")


class InferenceOutput(BaseModel):
    """Output of inference step → input for eval step."""

    features_path: str = Field(..., description="Path to output features")
    num_samples: int = Field(..., description="Total samples processed")
    checkpoint_ref: str = Field(..., description="W&B ref of checkpoint used")


class EvalInput(BaseModel):
    """Input for eval step (from InferenceOutput)."""

    features_path: str = Field(..., description="Path to features from inference")
    ground_truth_path: str = Field(
        default="/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_2",
        description="Path to ground truth data",
    )
    split_path: str = Field(
        default="/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json",
        description="Path to split JSON",
    )
    task_id: str = Field(default="virtual_map", description="VCB task id ('virtual_map' or 'phenorescue')")
    split_index: int = Field(default=0, description="Fold index to evaluate")


class EvalOutput(BaseModel):
    """Output of eval step."""

    metrics: dict[str, float] = Field(..., description="Evaluation metrics")
    features_path: str = Field(..., description="Features evaluated")
    eval_type: str = Field(default="vcb", description="Evaluation type")
