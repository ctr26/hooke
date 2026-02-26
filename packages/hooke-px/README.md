# hooke-forge

Train Diffusion Transformers on phenomics data using flow matching (linear interpolant). Intended to be simple for integration with the rest of Hooke.

## Installation

```bash
pip install -e .
# or
uv pip install -e .
```

The package lives in `src/hooke_forge/` and follows a standard `src`-layout.

## Package structure

```
src/hooke_forge/
├── model/                  # Core model components
│   ├── layers.py           # Attention, Transformer building blocks
│   ├── architecture.py     # DiT, ConditionedMLP, get_model_cls
│   ├── flow_matching.py    # JointFlowMatching, get_model factory
│   ├── context_encoders.py # TransformerEncoder, ScalarEmbedder
│   └── tokenizer.py        # DataFrameTokenizer, MetaDataConfig
├── training/               # Training loop and entry points
│   ├── state.py            # TrainState, log(), set_metrics_log_path()
│   ├── trainer.py          # train() — main training loop
│   ├── train.py            # SLURM launcher for flow matching training
│   └── train_tx_ae.py      # SLURM launcher for Tx autoencoder training
├── evaluation/             # Metrics and evaluation routines
│   ├── metrics.py          # FD, FD-sqrtm, FD-∞, cosine sim, PRDC, energy distance, retrieval
│   ├── px_metrics.py       # evaluate_px, visualise_phenomics, compute_phenomics_metrics
│   ├── tx_metrics.py       # evaluate_tx
│   └── eval.py             # Offline checkpoint evaluation with SLURM launcher
├── data/                   # Dataset and data loading
│   └── dataset.py          # CellDataset, TxDataset, get_dataloaders
├── utils/                  # Shared utilities
│   ├── distributed.py      # Distributed context manager, rank_zero
│   ├── ema.py              # KarrasEMA
│   ├── encoders.py         # DINOv2Detector, Phenom2Detector, StabilityCPEncoder
│   ├── infinite_dataloader.py
│   ├── name_run.py
│   └── profiler.py
└── inference/              # Distributed inference pipeline
    ├── checkpoint.py       # find_checkpoint, extract_model_config
    ├── distributed.py      # run_distributed_inference orchestration
    ├── lineage.py          # Model lineage tracing
    ├── prepare_eval.py     # prepare_for_vcb
    ├── validation.py       # check_completion, recover_completion_status
    ├── vcb_datasets.py     # VCB dataset configs and transforms
    ├── run.py              # Unified inference CLI (checkpoint → VCB-ready)
    ├── run_distributed.py  # Distributed inference master script
    └── run_worker.py       # Per-SLURM-node inference worker
```

## Entry points

After `pip install -e .`, the following console scripts are available:

| Command | Description |
|---|---|
| `hooke-train` | Launch training via SLURM or locally |
| `hooke-eval` | Evaluate saved checkpoints |
| `hooke-infer` | Run distributed inference |

Or invoke modules directly:

```bash
# Training
python -m hooke_forge.training.train --help

# Offline evaluation
python -m hooke_forge.evaluation.eval --help

# Full inference pipeline (checkpoint → VCB-ready)
python -m hooke_forge.inference.run --help

# Distributed inference master
python -m hooke_forge.inference.run_distributed --help
```

## Configuration

This repo uses [`ornamentalist`](https://github.com/valencelabs/ornamentalist) to auto-generate a CLI from function signatures. Configurable parameters are annotated with `ornamentalist.Configurable[default]` and exposed as `--group.param value` flags. If you are integrating with a project that does not use `ornamentalist`, the decorator can be removed with minor changes to function signatures.

## Training example

```bash
python -m hooke_forge.training.train \
    --flow_model.modality joint \
    --flow_model.tx_feature_dim 1024 \
    --launcher.cluster slurm \
    --launcher.nodes 4 \
    --launcher.gpus 8
```

## Inference example

```bash
python -m hooke_forge.inference.run \
    --training-dir outputs/1768305605/12583183 \
    --step 200000 \
    --dataset /path/to/metadata.parquet \
    --output-base /path/to/metrics \
    --ground-truth-dir /path/to/vcb_ground_truth \
    --task-id phenorescue
```
