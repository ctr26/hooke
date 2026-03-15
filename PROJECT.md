# Hooke - Project Specification

## Problem

Engineering standards and agent configurations are scattered across projects, leading to inconsistent tooling, CI setup, and code patterns. New projects start from scratch instead of inheriting proven defaults.

## Solution

Hooke is the single source of truth for engineering standards. It provides template files, agent rules, and documentation that new projects copy or reference.

## What Hooke Contains

### Standards (the 6-file standard)

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `CONTRIBUTING.md` | PR process, git workflow, commit conventions |
| `GUIDELINES.md` | Code patterns, anti-patterns, do/don't |
| `STANDARDS.md` | Tooling: uv, ruff, ty, pytest, versions |
| `AGENTS.md` | Agent/workflow rules, model selection, safety |
| `PROJECT.md` | This file — meta-spec for the standards repo |

### Templates (`templates/`)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python project config with ruff, pytest, ty |
| `Makefile` | Dev commands: install, test, lint, prepush |
| `.github/workflows/ci.yml` | CI pipeline: lint, format, typecheck, test |
| `PROJECT_SPEC.md` | Architecture spec template for new projects |

### Agent Rules

| Location | Purpose |
|----------|---------|
| `.claude/rules/` | Language and domain-specific rules |
| `.cursor/rules/` | Mirror of `.claude/rules/` for Cursor IDE |

## Usage

### New Project

```bash
gh repo create ctr26/<name> --private --clone && cd <name>

# Copy standards
cp ~/projects/ctr26/hooke/AGENTS.md .
cp ~/projects/ctr26/hooke/templates/pyproject.toml .
cp ~/projects/ctr26/hooke/templates/Makefile .
cp -r ~/projects/ctr26/hooke/templates/.github .
cp ~/projects/ctr26/hooke/templates/PROJECT_SPEC.md .
cp -r ~/projects/ctr26/hooke/.claude .
cp -r ~/projects/ctr26/hooke/.cursor .

# Edit PROJECT_SPEC.md, pyproject.toml for your project
# First commit
git add -A && git commit -m "chore: scaffold from hooke"
```

### Existing Project

Copy individual files as needed, or add hooke as a submodule.

## Non-Goals

- Hooke is not a cookiecutter/copier template generator
- No runtime code — standards and configs only
- No project-specific logic

## Maintenance

- Update standards when patterns are validated across 2+ projects
- Templates track the latest tooling versions
- Agent rules evolve via the self-improvement protocol in AGENTS.md
