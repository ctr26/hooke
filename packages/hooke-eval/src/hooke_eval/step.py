"""Eval step with Weave lineage.

Consumes InferenceOutput from hooke-px via weave.ref().
Runs VCB evaluation and returns metrics.
"""

import logging
from pathlib import Path

import weave

from hooke_px.schemas import EvalInput, EvalOutput, InferenceOutput

log = logging.getLogger(__name__)


def _run_vcb_evaluation(
    features_path: str,
    ground_truth_path: str,
    split_path: str,
    task_id: str = "virtual_map",
    split_index: int = 0,
) -> dict[str, float]:
    """Run VCB evaluation and return metrics as a flat dict.

    Args:
        features_path: Path to prediction features directory
        ground_truth_path: Path to ground truth dataset directory
        split_path: Path to split JSON
        task_id: VCB task ("phenorescue" or "virtual_map")
        split_index: Which fold to evaluate

    Returns:
        Dict of metric_name -> score (mean across groups)
    """
    import polars as pl
    from pydantic import TypeAdapter

    from vcb._cli.evaluate.px_cli import get_metric_suites_for_task_id
    from vcb.data_models.config import TASK_ADAPTERS_TYPE, EvaluationConfig
    from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
    from vcb.data_models.dataset.dataset_directory import DatasetDirectory
    from vcb.data_models.dataset.predictions import PredictionPaths

    # Load ground truth
    gt_dir = DatasetDirectory(root=Path(ground_truth_path))
    ground_truth = AnnotatedDataMatrix(**gt_dir.model_dump())

    # Load predictions
    pred_paths = PredictionPaths(root=Path(features_path))
    predictions = AnnotatedDataMatrix(
        **pred_paths.model_dump(),
        var_path=ground_truth.var_path,
        metadata_path=ground_truth.metadata_path,
    )

    # Build task adapter
    type_adapter = TypeAdapter(TASK_ADAPTERS_TYPE)
    task_adapter_kind = "drugscreen" if task_id == "phenorescue" else "singles"

    config = EvaluationConfig(
        ground_truth=type_adapter.validate_python(
            {"kind": task_adapter_kind, "dataset": ground_truth}
        ),
        predictions=type_adapter.validate_python(
            {"kind": task_adapter_kind, "dataset": predictions}
        ),
        split_path=Path(split_path),
        split_index=split_index,
        metric_suites=get_metric_suites_for_task_id(task_id, distributional_metrics=False),
    )

    results: pl.DataFrame = config.execute()

    # Aggregate to mean scores per metric
    summary = results.group_by("metric").agg(pl.col("score").mean().alias("mean"))
    return {row["metric"]: row["mean"] for row in summary.iter_rows(named=True)}


@weave.op()
def eval_step(input: EvalInput) -> EvalOutput:
    """Evaluate features using VCB metrics.

    Weave tracks lineage from inference -> eval.
    """
    log.info(f"Evaluating: {input.features_path}")
    log.info(f"Ground truth: {input.ground_truth_path}")
    log.info(f"Split: {input.split_path}")

    metrics = _run_vcb_evaluation(
        features_path=input.features_path,
        ground_truth_path=input.ground_truth_path,
        split_path=input.split_path,
        task_id=input.task_id,
        split_index=input.split_index,
    )

    log.info(f"Eval complete: {metrics}")

    return EvalOutput(
        metrics=metrics,
        features_path=input.features_path,
        eval_type="vcb",
    )


def eval_from_inference(inference_ref: str) -> EvalOutput:
    """Run eval from inference output ref.

    Example:
        eval_from_inference("hooke-px/inference-output:latest")
    """
    weave.init("hooke-eval")

    inference_output: InferenceOutput = weave.ref(inference_ref).get()

    eval_input = EvalInput(
        features_path=inference_output.features_path,
    )

    return eval_step(eval_input)
