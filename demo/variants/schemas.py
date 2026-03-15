"""Shared schemas for all pipeline variants."""

from pydantic import BaseModel


# -- Typed per-step confs (used by nested + pipe variants) --


class DataConf(BaseModel):
    split_file: str = "data/splits/default.json"
    output_dir: str = "outputs/splits"


class ConditioningConf(BaseModel):
    data: DataConf
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainConf(BaseModel):
    conditioning: ConditioningConf
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class FinetuningConf(BaseModel):
    pretrain: PretrainConf
    checkpoint_path: str
    step: int
    target_cell_type: str = "ARPE19"


class InferenceConf(BaseModel):
    finetuning: FinetuningConf
    checkpoint_path: str
    step: int


class EvalConf(BaseModel):
    inference: InferenceConf
    features_path: str
    num_samples: int


class ResultsConf(BaseModel):
    metrics: dict[str, float]


# -- Flat accumulator (used by accumulator + method chain variants) --


class PipelineState(BaseModel):
    # Data
    split_file: str = "data/splits/default.json"
    split_path: str | None = None
    train_compounds: list[str] | None = None
    val_compounds: list[str] | None = None
    test_compounds: list[str] | None = None

    # Conditioning
    cell_types: list[str] | None = None
    assay_types: list[str] | None = None
    vocab_size: int | None = None
    conditioning_path: str | None = None

    # Pretrain
    pretrain_checkpoint: str | None = None
    pretrain_step_count: int | None = None

    # Finetuning
    finetune_checkpoint: str | None = None
    finetune_step_count: int | None = None
    target_cell_type: str | None = None

    # Inference
    features_path: str | None = None
    num_samples: int | None = None

    # Eval
    metrics: dict[str, float] | None = None
