# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

hooke-forge is a machine learning project that trains Diffusion Transformers on phenomics data using flow matching (linear interpolant). The project follows a standard `src`-layout with the main package in `src/hooke_forge/`.

## Development Setup

This project uses `uv` for Python environment management. Activate the environment with:

```bash
.venv/bin/activate
```

Install the package in development mode:

```bash
uv pip install -e .
```

## Core Architecture

The project is organized into several key modules:

- **model/**: Core model components including DiT architecture, flow matching, context encoders, and tokenizers
- **training/**: Training orchestration with SLURM integration via submitit
- **evaluation/**: Offline checkpoint evaluation on various datasets
- **data/**: Dataset loading for cell and treatment (Tx) data
- **inference/**: Distributed inference pipeline with SLURM-based distributed execution
- **utils/**: Shared utilities for distributed training, EMA, evaluation metrics, and profiling

The core model is `JointFlowMatching` which supports multiple modalities (images and features) with shared context encoding and time embedding.

## Common Commands

### Training
```bash
# Direct module invocation
python -m hooke_forge.training.train --help

# Console script (after installation)
hooke-train --flow_model.modality joint --launcher.cluster slurm --launcher.nodes 4
```

### Evaluation
```bash
# Evaluate saved checkpoints
python -m hooke_forge.evaluation.eval --help
hooke-eval
```

### Inference
```bash
# Run distributed inference
python -m hooke_forge.inference.run_distributed --help
hooke-infer

# Single-node inference
python -m hooke_forge.inference.run --help
```

### Helper Scripts
Several bash scripts in `scripts/` provide common workflows:
- `scripts/launch_eval.sh`: Launch distributed evaluation across multiple checkpoints
- `scripts/launch_inference.sh`: Launch inference jobs
- `scripts/test_adaptor_embedding.sh`: Test embedding components

## Configuration System

The project uses `ornamentalist` for automatic CLI generation from function signatures. Parameters are annotated with `ornamentalist.Configurable[default]` and exposed as `--group.param value` flags. This allows for hierarchical configuration without separate config files.

Example usage patterns:
- Flow model parameters: `--flow_model.modality joint --flow_model.tx_feature_dim 1024`
- Launcher settings: `--launcher.cluster slurm --launcher.nodes 4 --launcher.gpus 8`
- Data parameters: `--data.batch_size 32 --data.num_workers 8`

## Development Workflow

1. **Local Development**: Use `uv run` for running modules directly
2. **SLURM Cluster**: The training system integrates with SLURM via submitit for distributed training
3. **Experiment Tracking**: Uses Weights & Biases (wandb) for logging and tracking
4. **Output Structure**: Training runs create numbered output directories in `outputs/` with checkpoints and logs

## Key Dependencies

- **torch**: Core deep learning framework
- **diffusers**: Diffusion model utilities
- **accelerate**: Distributed training acceleration
- **submitit**: SLURM job submission
- **ornamentalist**: Configuration management
- **wandb**: Experiment tracking
- **zarr**: Data storage format
- **timm**: Vision model components

## Testing

No formal test suite exists. Testing is done via standalone scripts in `scripts/` directory, particularly `test_adaptor_embedding.sh` for component testing.