# Schema-Governed Pipeline with W&B Lineage

## Core Pattern

**Each step returns the next step's input schema.**

```
conditioning_step() → PretrainInput
pretrain_step(PretrainInput) → FinetuneInput
finetune_step(FinetuneInput) → EvalInput
eval_step(EvalInput) → Result
```

## Files

| File | Purpose |
|------|---------|
| `schema_step.py` | Core pattern: schemas + @step decorator + pipeline |
| `artifacts.py` | Low-level W&B artifact helpers |

## Quick Start

```bash
python demo/lineage/schema_step.py
```

## Example

```python
class FinetuneInput(BaseModel):
    """pretrain_step returns this."""
    checkpoint_path: str
    cell_types: list[str]
    step: int

@step(artifact_type="model")
def pretrain_step(input: PretrainInput, output_dir: Path) -> FinetuneInput:
    return FinetuneInput(
        checkpoint_path=str(ckpt),
        cell_types=input.cell_types,
        step=200000,
    )

# Pipeline: direct typed chaining
pretrain_in: PretrainInput = conditioning_step(...)
finetune_in: FinetuneInput = pretrain_step(pretrain_in, ...)
```

Output IS next input. No dict validation needed.
