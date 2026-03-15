"""End-to-end pipeline: inference -> eval with Weave lineage.

Usage:
    python -m hooke_px.pipeline \
        --checkpoint /path/to/checkpoint.ckpt \
        --dataset /path/to/metadata.parquet \
        --output-dir ./outputs/pipeline_run

Weave tracks full lineage: which checkpoint produced which features,
and which features produced which metrics.
"""

import argparse
import logging

import weave

from hooke_px.inference.step import inference_step
from hooke_px.schemas import EvalInput, InferenceInput

log = logging.getLogger(__name__)


@weave.op()
def run_pipeline(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    ground_truth_path: str = "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_2",
    split_path: str = "/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json",
    batch_size: int = 3,
    num_workers: int = 100,
    num_samples: int = 36,
    partition: str = "hopper",
    gpus_per_node: int = 4,
    weave_project: str = "hooke-px",
) -> dict:
    """Run full pipeline: inference -> eval.

    Weave tracks lineage across both steps.

    Returns:
        Dict with inference output and eval metrics.
    """
    from hooke_eval.step import eval_step

    # Step 1: Inference
    log.info("=" * 40)
    log.info("Step 1: Inference")
    log.info("=" * 40)

    inference_input = InferenceInput(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=num_samples,
        partition=partition,
        gpus_per_node=gpus_per_node,
    )
    inference_output = inference_step(inference_input)
    log.info(f"Inference done: {inference_output.features_path} ({inference_output.num_samples} samples)")

    # Step 2: Eval
    log.info("=" * 40)
    log.info("Step 2: Evaluation")
    log.info("=" * 40)

    eval_input = EvalInput(
        features_path=inference_output.features_path,
        ground_truth_path=ground_truth_path,
        split_path=split_path,
    )
    eval_output = eval_step(eval_input)
    log.info(f"Eval done: {eval_output.metrics}")

    return {
        "features_path": inference_output.features_path,
        "num_samples": inference_output.num_samples,
        "checkpoint_ref": inference_output.checkpoint_ref,
        "metrics": eval_output.metrics,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run inference -> eval pipeline with Weave lineage")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, W&B artifact, or Weave ref")
    parser.add_argument("--dataset", required=True, help="Input metadata parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ground-truth", default="/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_2")
    parser.add_argument("--split", default="/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=36)
    parser.add_argument("--partition", default="hopper")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--weave-project", default="hooke-px")
    args = parser.parse_args()

    weave.init(args.weave_project)

    result = run_pipeline(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        split_path=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        partition=args.partition,
        gpus_per_node=args.gpus_per_node,
        weave_project=args.weave_project,
    )

    print(f"\nPipeline complete!")
    print(f"  Features: {result['features_path']}")
    print(f"  Samples: {result['num_samples']}")
    print(f"  Checkpoint: {result['checkpoint_ref']}")
    print(f"  Metrics: {result['metrics']}")
    print(f"\nView lineage: https://wandb.ai/<entity>/{args.weave_project}/weave")


if __name__ == "__main__":
    main()
