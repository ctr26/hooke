# Schema-Governed Pipeline with W&B Lineage

## Core Pattern

**Each step returns the next step's input schema.**

Splits and conditioning are tracked as artifacts.

```
splits_step() → ConditioningInput
conditioning_step(ConditioningInput) → PretrainInput
pretrain_step(PretrainInput) → FinetuneInput
finetune_step(FinetuneInput) → EvalInput
eval_step(EvalInput) → Result
```

## Quick Start

```bash
python demo/lineage/schema_step.py
```

## Example

```python
class ConditioningInput(BaseModel):
    """splits_step returns this."""
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]

class PretrainInput(BaseModel):
    """conditioning_step returns this."""
    split_path: str
    train_compounds: list[str]
    test_compounds: list[str]
    cell_types: list[str]
    vocab_size: int

@step(artifact_type="split")
def splits_step(...) -> ConditioningInput:
    return ConditioningInput(...)

@step(artifact_type="config")
def conditioning_step(input: ConditioningInput, ...) -> PretrainInput:
    return PretrainInput(
        split_path=input.split_path,  # Carry forward
        train_compounds=input.train_compounds,
        test_compounds=input.test_compounds,
        cell_types=["ARPE19", "HUVEC"],
        vocab_size=2048,
    )
```

## W&B Lineage

Change splits or conditioning → W&B tracks which checkpoints are stale.

## Files

| File | Purpose |
|------|---------|
| `schema_step.py` | Full pipeline with splits + conditioning |
| `artifacts.py` | W&B artifact helpers |
