# HookeTx
Repository for Transcriptomics perturbation prediction

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10–3.12 (PyG wheels do not support 3.13 yet).

```bash
# Install dependencies into .venv
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

## Running an experiment (trek_drugscreen template)

From the repo root:

```bash
python main.py
```

The app uses Hydra with the `configs/templates/trek_drugscreen/cfg.yaml` template. Override any option from the CLI, e.g.:

```bash
python main.py trainer.max_epochs=1 trainer.batch_size_train=16
```
