"""Shared schemas for all pipeline variants.

Schemas carry actual outputs (lists, dicts, embeddings) not file paths.
"""

from pydantic import BaseModel


# -- Typed per-step confs (used by nested, pipe, weave_refs variants) --


class DataConf(BaseModel):
    split_file: str = "data/splits/default.json"


class ConditioningConf(BaseModel):
    data: DataConf
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class PretrainConf(BaseModel):
    conditioning: ConditioningConf
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_weights: list[float]


class FinetuningConf(BaseModel):
    pretrain: PretrainConf
    model_weights: list[float]
    step: int
    target_cell_type: str = "ARPE19"


class InferenceConf(BaseModel):
    finetuning: FinetuningConf
    model_weights: list[float]
    step: int


class EvalConf(BaseModel):
    inference: InferenceConf
    features: list[list[float]]
    num_samples: int


class ResultsConf(BaseModel):
    metrics: dict[str, float]


# -- Flat accumulator (used by accumulator, method_chain, pipe variants) --


class PipelineState(BaseModel):
    # Data
    split_file: str = "data/splits/default.json"
    train_compounds: list[str] | None = None
    val_compounds: list[str] | None = None
    test_compounds: list[str] | None = None

    # Conditioning
    cell_types: list[str] | None = None
    assay_types: list[str] | None = None
    vocab_size: int | None = None
    conditioning_weights: list[float] | None = None

    # Pretrain
    model_weights: list[float] | None = None
    pretrain_step: int | None = None

    # Finetuning
    finetuned_weights: list[float] | None = None
    finetune_step: int | None = None
    target_cell_type: str | None = None

    # Inference
    features: list[list[float]] | None = None
    num_samples: int | None = None

    # Eval
    metrics: dict[str, float] | None = None
