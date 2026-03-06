#!/usr/bin/env python3
"""Inference pipeline: checkpoint -> distributed inference -> validation -> map building.

Usage:
    python -m hooke_forge.inference.run \
        --training-dir outputs/1768305605/12583183 \
        --step 200000 \
        --dataset /path/to/metadata.parquet \
        --output-base /path/to/metrics \
        --build-maps
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
    parser = argparse.ArgumentParser(description="Run inference pipeline from checkpoint to map building")
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
        required=True,
        help="Input parquet with observations",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        required=True,
        help="Base output directory (creates {data_version}/{model_config}/step_{N}/ subdirectory)",
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
        help="Skip inference, only run validation and map building",
    )
    parser.add_argument(
        "--build-maps",
        action="store_true",
        help="Build perturbation similarity maps after inference",
    )
    parser.add_argument(
        "--perturbation-col",
        type=str,
        default="inchikey",
        help="Column for perturbation grouping in map building (default: inchikey)",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=None,
        help="Representations to extract (default: auto-detect from modality)",
    )
    parser.add_argument(
        "--tx-zarr-path",
        type=str,
        default="",
        help="Path to tx feature zarr (required for tx modality)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow imports when just checking --help
    from hooke_forge.inference.checkpoint import extract_model_config, find_checkpoint
    from hooke_forge.inference.distributed import run_distributed_inference
    from hooke_forge.inference.lineage import get_model_lineage
    from hooke_forge.inference.validation import (
        check_completion,
        recover_completion_status,
    )

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

    output_dir = args.output_base / lineage["data_version"] / lineage["model_config"] / f"step_{args.step}"
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
            representations=args.representations,
            tx_zarr_path=args.tx_zarr_path,
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

    # Step 5: Build maps
    if args.build_maps:
        log.info("=" * 60)
        log.info("Step 5: Building perturbation similarity maps")
        log.info("=" * 60)

        from hooke_forge.evaluation.map_building import build_maps_from_inference

        maps = build_maps_from_inference(
            output_dir=output_dir,
            perturbation_cols=[args.perturbation_col],
        )
        log.info(f"Built {len(maps)} maps")

    log.info("=" * 60)
    log.info("Complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
