# HookeTx
Repository for Transcriptomics perturbation prediction

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10–3.12 (PyG wheels do not support 3.13 yet).

```bash
# Install dependencies into .venv
uv sync

# Optional: include dev tools (pytest, ruff, mypy, pre-commit)
uv sync --extra dev
```

Activate the environment:

```bash
source .venv/bin/activate
```

Or run commands without activating: `uv run python main.py`, `uv run ipython`, etc.
