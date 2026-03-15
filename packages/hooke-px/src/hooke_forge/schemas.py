"""Schema-governed pipeline: each step returns the next step's input schema.

splits_step() → SplitsOutput
conditioning_step(SplitsOutput) → ConditioningOutput
pretrain_step(ConditioningOutput) → PretrainOutput
finetune_step(PretrainOutput) → EvalInput
eval_step(EvalInput) → Result

inference_step(InferenceInput) → InferenceOutput
"""

from pydantic import BaseModel


class SplitsOutput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]


class ConditioningOutput(BaseModel):
    split_path: str
    train_compounds: list[str]
    val_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    assay_types: list[str]
    vocab_size: int
    conditioning_path: str


class PretrainOutput(BaseModel):
    checkpoint_path: str
    cell_types: list[str]
    vocab_size: int
    step: int
    test_compounds: list[str]


class EvalInput(BaseModel):
    checkpoint_path: str
    target_cell_type: str
    test_compounds: list[str]


class InferenceInput(BaseModel):
    checkpoint_path: str
    dataset_path: str
    batch_size: int = 32
    num_workers: int = 4


class InferenceOutput(BaseModel):
    features_path: str
    num_samples: int
