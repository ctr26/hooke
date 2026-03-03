#!/usr/bin/env python3
"""Unified inference pipeline for big-img model evaluation.

Combines checkpoint discovery, distributed inference, validation, and VCB
preparation into a single workflow.

Usage:
    python -m hooke_forge.inference.run \
        --training-dir outputs/1768305605/12583183 \
        --step 200000 \
        --dataset /path/to/metadata.parquet \
        --output-base /path/to/metrics \
        --ground-truth-dir /path/to/vcb_ground_truth \
        --task-id phenorescue
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference pipeline from checkpoint to VCB-ready data")
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="Training output directory containing checkpoints/",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Checkpoint step number to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Input parquet with observations (optional if --vcb-dataset provided)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        required=True,
        help="Base output directory (creates step_{N}/ subdirectory)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="VCB ground truth dataset directory",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        choices=["virtual_map", "phenorescue"],
        help="VCB task (determines biological_context)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=100,
        help="Number of SLURM workers (default: 100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=36,
        help="Samples per image for predictions (default: 36)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size per worker (default: 3)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="hopper",
        help="SLURM partition (default: hopper)",
    )
    parser.add_argument(
        "--qos",
        type=str,
        default=None,
        help="SLURM QOS (default: None)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference, only run validation and VCB prep",
    )
    parser.add_argument(
        "--vcb-dataset",
        type=str,
        choices=["drugscreen", "cross_cell_line"],
        default=None,
        help="VCB dataset type (transforms obs parquet to inference format)",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Include validation observations in predictions (may cause VCB mismatches)",
    )
    parser.add_argument(
        "--create-test-only",
        action="store_true",
        help="Force creation of test-only predictions (default when test+valid detected)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow imports when just checking --help
    from hooke_forge.inference.checkpoint import extract_model_config, find_checkpoint
    from hooke_forge.inference.distributed import run_distributed_inference
    from hooke_forge.inference.lineage import get_model_lineage
    from hooke_forge.inference.prepare_eval import prepare_for_vcb, print_vcb_command
    from hooke_forge.inference.validation import check_completion, recover_completion_status
    from hooke_forge.inference.vcb_datasets import get_vcb_obs_path

    # Resolve dataset path
    if args.dataset is None and args.vcb_dataset is None:
        log.error("Must provide either --dataset or --vcb-dataset")
        sys.exit(1)

    if args.dataset is None:
        # Use default VCB obs path
        args.dataset = Path(get_vcb_obs_path(args.vcb_dataset))
        log.info(f"Using VCB dataset: {args.vcb_dataset}")
        log.info(f"  obs path: {args.dataset}")

    # Step 1: Find checkpoint
    log.info("=" * 60)
    log.info("Step 1: Finding checkpoint")
    log.info("=" * 60)

    checkpoint = find_checkpoint(args.training_dir, args.step)
    log.info(f"Found checkpoint: {checkpoint}")

    model_config = extract_model_config(args.training_dir)
    if model_config:
        log.info(f"Model config: {model_config}")

    # Step 2: Trace model lineage and set up output directory
    log.info("=" * 60)
    log.info("Step 2: Tracing model lineage")
    log.info("=" * 60)

    lineage = get_model_lineage(args.training_dir)
    log.info(f"Data version: {lineage['data_version']}")
    log.info(f"Model architecture: {lineage['model_config']}")
    log.info(f"Lineage depth: {len(lineage['lineage_chain'])} training runs")

    # Construct structured output path:
    # {base_path}/{data_version}/{task}/{model_config}/step_{step}
    output_dir = (
        args.output_base / lineage["data_version"] / args.task_id / lineage["model_config"] / f"step_{args.step}"
    )
    log.info(f"Output directory: {output_dir}")

    # Step 3: Run distributed inference
    if not args.skip_inference:
        log.info("=" * 60)
        log.info("Step 3: Running distributed inference")
        log.info("=" * 60)

        run_distributed_inference(
            checkpoint_path=checkpoint,
            input_parquet=args.dataset,
            output_dir=output_dir,
            model_config=model_config,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            partition=args.partition,
            qos=args.qos,
            vcb_dataset=args.vcb_dataset,
        )
    else:
        log.info("Skipping inference (--skip-inference)")

    # Step 4: Validate completion
    log.info("=" * 60)
    log.info("Step 4: Validating completion")
    log.info("=" * 60)

    complete, total = check_completion(output_dir)
    log.info(f"Completion: {complete}/{total}")

    if complete < total:
        log.info("Recovering completion status from zarr...")
        recovered = recover_completion_status(output_dir)
        log.info(f"Recovered {recovered} rows")

        complete, total = check_completion(output_dir)
        log.info(f"After recovery: {complete}/{total}")

        if complete < total:
            log.error(f"Incomplete inference: {complete}/{total}")
            log.error("Re-run without --skip-inference to complete remaining rows")
            sys.exit(1)

    # Step 5: Prepare for VCB
    log.info("=" * 60)
    log.info("Step 5: Preparing for VCB evaluation")
    log.info("=" * 60)

    eval_dir = output_dir / "eval"

    # Check splits to determine what to create
    try:
        import polars as pl

        pred_metadata = pl.read_parquet(output_dir / "prepared_metadata.parquet")
        split_values = pred_metadata["split"].unique().to_list()
        has_test = any(s.startswith("test") for s in split_values)
        has_valid = any(s.startswith("valid") for s in split_values)
    except Exception as e:
        log.warning(f"Could not check observation splits: {e}")
        has_test = has_valid = False

    both_splits_exist = has_test and has_valid

    if both_splits_exist and args.include_validation:
        # Create both test-only and with-validation predictions
        log.info("Creating both test-only and with-validation predictions")

        # Test-only (default/recommended)
        prepare_for_vcb(
            predictions_dir=output_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=eval_dir,
            task_id=args.task_id,
            lineage=lineage,
            test_only=True,
        )

        # With validation (user requested)
        prepare_for_vcb(
            predictions_dir=output_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=eval_dir,
            task_id=args.task_id,
            lineage=lineage,
            test_only=False,
        )

        # Print next steps
        log.info("=" * 60)
        log.info("Complete!")
        log.info("=" * 60)

        print_vcb_command(
            eval_dir,
            args.ground_truth_dir,
            args.task_id,
            pred_dir_name="predictions",
            split_file_name="split.json",
            note=" (RECOMMENDED - test-only)",
        )

        print_vcb_command(
            eval_dir,
            args.ground_truth_dir,
            args.task_id,
            pred_dir_name="predictions_with_valid",
            split_file_name="split_with_valid.json",
            note=" (with validation - may cause mismatches)",
        )

    else:
        # Create single set of predictions
        if both_splits_exist:
            # Default: test-only
            test_only = True
            log.info("Creating test-only predictions (default - VCB expects test-only)")
            log.info("  Use --include-validation to also create predictions with validation")
        else:
            # Only one type of split available
            test_only = False
            if has_test:
                log.info("Only test observations found")
            elif has_valid:
                log.info("Only validation observations found")
            else:
                log.info("No test/validation splits found - using all observations")

        prepare_for_vcb(
            predictions_dir=output_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=eval_dir,
            task_id=args.task_id,
            lineage=lineage,
            test_only=test_only,
        )

        # Print next steps
        log.info("=" * 60)
        log.info("Complete!")
        log.info("=" * 60)

        pred_dir = "predictions" if test_only else "predictions_with_valid"
        split_file = "split.json" if test_only else "split_with_valid.json"

        print_vcb_command(
            eval_dir, args.ground_truth_dir, args.task_id, pred_dir_name=pred_dir, split_file_name=split_file
        )


if __name__ == "__main__":
    main()
