"""Model, data, and pipeline configuration schemas."""

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model_class: str
    weights_path: str
    device: str = "cuda"
    precision: str = "float32"


class DataConfig(BaseModel):
    dataset_path: str
    batch_size: int = 32
    num_workers: int = 4
    preprocessing_steps: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    output_dir: str = "outputs"
    project: str = "hooke-px"
    cache_dir: str = "/data/valence/cache"
