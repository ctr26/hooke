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

```bash
# New project from hooke
gh repo create ctr26/<name> --private --clone && cd <name>
cp ~/projects/ctr26/hooke/AGENTS.md .
cp ~/projects/ctr26/hooke/templates/pyproject.toml .
cp ~/projects/ctr26/hooke/templates/Makefile .
cp -r ~/projects/ctr26/hooke/templates/.github .
cp ~/projects/ctr26/hooke/templates/PROJECT_SPEC.md .
cp -r ~/projects/ctr26/hooke/.claude .
cp -r ~/projects/ctr26/hooke/.cursor .
```

## Agent Configuration

| Location | Purpose |
|----------|---------|
| `.claude/rules/` | Language and domain rules (Python, TS, ML, engineering) |
| `.cursor/rules/` | Mirror of `.claude/rules/` for Cursor IDE |

## License

MIT
