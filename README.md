# Hooke

ML models for biological perturbation prediction.

## Engineering Standards

This repo follows the hooke 6-file standard:

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CONTRIBUTING.md` | PR process, git workflow, commit conventions |
| `GUIDELINES.md` | Code patterns and anti-patterns |
| `STANDARDS.md` | Tooling: uv, ruff, ty, pytest |
| `AGENTS.md` | Agent rules, model selection, safety |
| `PROJECT.md` | Architecture spec |

## Templates

Ready-to-copy project scaffolding in `templates/`:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python config with ruff, pytest, ty, hatchling |
| `Makefile` | Dev commands: install, test, lint, prepush |
| `.github/workflows/ci.yml` | CI: lint, format, typecheck, test, coverage |
| `PROJECT_SPEC.md` | Architecture spec template |

## Quick Start

```python
from hooke.pipeline import Pipeline

pipe = Pipeline.from_config("configs/example.yaml")
result = pipe({"features": [0.5, -1.2, 0.3]})
print(result["predictions"])
```

```bash
# Install
uv sync --all-extras

# Run tests
make test

# Lint + format
make lint && make format
```

## Agent Configuration

| Location | Purpose |
|----------|---------|
| `.claude/rules/` | Language and domain rules (Python, TS, ML, engineering) |
| `.cursor/rules/` | Mirror of `.claude/rules/` for Cursor IDE |

## License

MIT
