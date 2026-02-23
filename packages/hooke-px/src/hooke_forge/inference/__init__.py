"""Unified inference module for big-img model evaluation.

This module provides a complete pipeline from trained checkpoint to VCB-ready
evaluation data:

1. checkpoint - Discover and load checkpoints from training output
2. distributed - Run distributed inference on SLURM
3. validation - Validate inference completion
4. prepare_eval - Prepare predictions for VCB evaluation

The pipeline automatically creates test-only predictions by default when both test
and validation observations are present, since VCB typically filters ground truth
to test observations only.

Usage:
    python -m inference.run \
        --training-dir outputs/1768305605/12583183 \
        --step 200000 \
        --dataset /path/to/metadata.parquet \
        --output-base /path/to/metrics \
        --ground-truth-dir /path/to/vcb_ground_truth \
        --task-id phenorescue

Options:
    --include-validation    Include validation observations (may cause VCB mismatches)
    --create-test-only     Force test-only predictions (default when test+valid detected)
"""

from hooke_forge.inference.checkpoint import find_checkpoint, extract_model_config
from hooke_forge.inference.validation import check_completion, recover_completion_status
from hooke_forge.inference.prepare_eval import prepare_for_vcb
from hooke_forge.inference.lineage import get_model_lineage, parse_config_from_log
from hooke_forge.inference.vcb_datasets import (
    VCB_DATASETS,
    transform_vcb_dataset,
    get_vcb_dataset_types,
    get_vcb_obs_path,
)

__all__ = [
    "find_checkpoint",
    "extract_model_config",
    "check_completion",
    "recover_completion_status",
    "prepare_for_vcb",
    "get_model_lineage",
    "parse_config_from_log",
    "VCB_DATASETS",
    "transform_vcb_dataset",
    "get_vcb_dataset_types",
    "get_vcb_obs_path",
]
