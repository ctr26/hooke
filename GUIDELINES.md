# Guidelines

Code patterns and anti-patterns for hooke-standard projects.

## Architecture

### Do

- **Cheap**: Minimize hosting costs, use free tiers
- **Scalable**: Cloud-first, horizontal scaling
- **Low LOC**: Minimal code, maximum impact
- **Minimal deps**: Fewer dependencies = fewer vulnerabilities
- **Schemas at boundaries**: Pydantic for data validation, types for internal contracts

### Don't

- Over-engineer: no premature abstractions, no helpers for one-time operations
- Add features beyond what's asked: a bug fix doesn't need surrounding cleanup
- Design for hypothetical futures: three similar lines > premature abstraction
- Add backwards-compatibility shims when you can just change the code

## Python Patterns

### Do

```python
# Type hints on public APIs
def train(config: TrainConfig) -> TrainResult: ...

# Pydantic for data boundaries
class ExperimentConfig(BaseModel):
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)

# Explicit over implicit
from pathlib import Path
output_dir = Path(config.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
```

### Don't

```python
# No bare dicts for structured data
config = {"lr": 0.001, "bs": 32}  # Use Pydantic instead

# No star imports
from module import *

# No mutable default arguments
def f(items=[]):  # Use items=None, then items = items or []
```

## ML & Research

### Do

- Baselines first: establish floor before adding complexity
- Log everything: hyperparams, metrics, git SHA, data version
- Checkpointing: save state every N steps, make resumable
- Fail early: validate config and data before GPU allocation
- Small-scale pilots: test on 1% of data first

### Don't

- Skip baselines: no new architecture without baseline comparison
- Train without tracking: every run must be in W&B
- Assume QoS: any job can be killed at any time
- Hardcode paths: use configs, not magic strings

## Safety

### Do

- `--dry-run` for cleanup operations
- Explicit user confirmation for destructive actions
- Log intent before executing
- Archive over delete

### Don't

- Auto-delete anything without confirmation
- Force push to protected branches
- Skip CI checks
- Commit secrets or credentials

## Testing

### Do

- Unit tests for core logic (80%+ coverage)
- Smoke tests for ML: overfit on tiny batch (loss < 0.1)
- Fixtures in `conftest.py`
- Mark slow tests: `@pytest.mark.slow`

### Don't

- Delete failing tests to make CI pass
- Write tests that depend on external services without mocks
- Skip tests in CI
