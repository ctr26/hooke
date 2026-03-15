# Standards

Tooling and versions for hooke-standard projects.

## Python

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Runtime | 3.11+ (apps), 3.9+ (libraries) |
| `uv` | Package manager | latest |
| `ruff` | Linter + formatter | >= 0.1.0 |
| `ty` | Type checker (Astral) | >= 0.1.0 |
| `pytest` | Testing | >= 8.0.0 |
| `pytest-cov` | Coverage | >= 4.1.0 |
| `hatchling` | Build backend | latest |

## Project Structure

```
<project>/
├── src/<package_name>/   # Source code (src layout)
│   ├── __init__.py
│   └── ...
├── tests/
│   ├── conftest.py
│   └── test_*.py
├── pyproject.toml        # Project config (single source)
├── Makefile              # Dev commands
├── PROJECT_SPEC.md       # Architecture spec
├── AGENTS.md             # Agent/workflow rules
└── .github/workflows/
    └── ci.yml            # CI pipeline
```

## Configuration

All tool config lives in `pyproject.toml`. No separate `.cfg`, `.ini`, or `.toml` files for tools.

### Ruff

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "SIM"]
ignore = ["E501"]
```

### Pytest

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--strict-markers", "--cov=src/<pkg>", "--cov-report=term-missing", "--tb=short"]
```

## Makefile Commands

| Command | Action |
|---------|--------|
| `make install` | `uv sync --all-extras` |
| `make test` | `uv run pytest` |
| `make test-fast` | `uv run pytest -m "not slow"` |
| `make lint` | `uv run ruff check .` |
| `make format` | `uv run ruff format .` |
| `make typecheck` | `uv run ty check src/` |
| `make prepush` | lint + typecheck + test-fast |

## CI Pipeline

Every PR runs:

1. `ruff check .` (lint)
2. `ruff format --check .` (format verification)
3. `ty check src/` (type checking)
4. `pytest --cov` (tests with coverage)

Coverage uploaded to Codecov.

## Quality Targets

| Metric | Target |
|--------|--------|
| Test coverage | 80%+ on core logic |
| Type hints | All public APIs |
| Lint | Zero warnings in CI |
| Format | Enforced by CI |

## ML-Specific Tooling

| Tool | Purpose |
|------|---------|
| `hydra-core` + `hydra-zen` | Configuration management |
| `wandb` | Experiment tracking |
| `deepspeed` | Distributed training |
| `safetensors` | Model serialization |
| `pydantic` | Schema validation |

## TypeScript (when applicable)

| Tool | Purpose | Version |
|------|---------|---------|
| Node | Runtime | 20+ (LTS) |
| TypeScript | Language | 5+ |
| `vitest` | Testing | latest |
| `eslint` | Linter | latest |
| `prettier` | Formatter | latest |
| `zod` | Runtime validation | latest |
