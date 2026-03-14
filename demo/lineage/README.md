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
| `weave_pipeline.py` | **W&B-native** — uses `@weave.op()` |
| `schema_step.py` | Custom `@step` decorator version |
| `artifacts.py` | W&B artifact helpers |

## Recommended: Weave

```python
import weave

weave.init("hooke")

@weave.op()
def pretrain_step(input: PretrainInput, output_dir: str) -> FinetuneInput:
    # Weave auto-tracks inputs, outputs, code version
    # Built-in caching — same inputs → cached result
    return FinetuneInput(...)
```

```bash
pip install weave
python demo/lineage/weave_pipeline.py
```
