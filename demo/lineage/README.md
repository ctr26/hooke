# Schema-Governed Pipeline with W&B Lineage

## Pattern

Each step has:
- **Pydantic schema** defining outputs
- **`@step` decorator** logging to W&B
- **Typed input** from previous step

## Files

| File | Purpose |
|------|---------|
| `schema_step.py` | Core pattern: schemas + decorator + pipeline |
| `artifacts.py` | Low-level W&B artifact helpers |
| `conditioning_example.py` | Example: conditioning update cascade |
| `pipeline_example.py` | Simple pipeline example |

## Quick Start

```bash
# Run schema-governed pipeline
python demo/lineage/schema_step.py

# Run conditioning update example
python demo/lineage/conditioning_example.py
```

## Schema Example

```python
class PretrainOutput(BaseModel):
    checkpoint_path: str
    conditioning: ConditioningOutput  # Previous step embedded
    step: int
    loss: float

@step(PretrainOutput, artifact_type="model")
def pretrain_step(conditioning: ConditioningOutput, ...) -> PretrainOutput:
    ...
```

## Flow

```
ConditioningOutput → PretrainOutput → FinetuneOutput
       ↓                   ↓                ↓
   W&B artifact       W&B artifact      W&B artifact
```

Each artifact contains schema metadata + embedded previous outputs.
