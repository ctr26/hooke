import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import zarr
from loguru import logger
from pydantic import TypeAdapter

from vcb._cli.evaluate.tx_cli import tx_evaluate_cli
from vcb.baselines import BASELINES
from vcb.data_models.config import TASK_ADAPTERS_TYPE
from vcb.data_models.dataset.anndata import TxAnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.split import Split
from vcb.preprocessing.steps.log1p import Log1pStep
from vcb.preprocessing.steps.match_genes import MatchGenesStep
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep
from vcb.settings import settings


def run_baseline_tx_cli(
    baseline_type: Annotated[
        str, typer.Option(..., "--baseline-type", "-b", help="type of baseline to evaluate")
    ],
    ground_truth_path: Annotated[
        Path, typer.Option(..., "--ground-truth-path", "-t", help="path to the ground truth directory")
    ],
    split_path: Annotated[Path, typer.Option(..., "--split-path", "-s", help="path to the split json file")],
    save_destination: Annotated[
        Path, typer.Option(..., "--save-destination", "-o", help="path to where results should be saved")
    ],
    predictions_var_path: Annotated[
        Path,
        typer.Option(..., "--predictions-var-path", "-v", help="path to the var file for the predictions"),
    ],
    task_id: Annotated[str, typer.Option(help="The task id. Either 'phenorescue' or 'virtual_map")],
    split_idx: Annotated[int, typer.Option(help="index of the split to evaluate")] = 0,
    predictions_gene_id_column: Annotated[
        str, typer.Option(help="column of the predictions to use for the ensembl gene id")
    ] = "ensembl_gene_id",
    ground_truth_gene_id_column: Annotated[
        str, typer.Option(help="column of the ground truth to use for the ensembl gene id")
    ] = "ensembl_gene_id",
    library_size: Annotated[
        int, typer.Option(help="library size to use for the evaluation (default ground truth median)")
    ] = None,
    distributional_metrics: Annotated[
        bool, typer.Option(help="whether to include distributional metrics (exclude to speed up evaluation)")
    ] = True,
    use_validation_split: Annotated[
        bool,
        typer.Option(
            help="whether to use the validation split instead of the test split (use to compare evaluation and fine tuning 1:1)"
        ),
    ] = False,
):
    """
    Evaluate baseline performance on a transcriptomics dataset.
    """

    # Update the global settings
    settings.save_dir = save_destination

    # Extract splits.
    split = Split.from_json(split_path)
    fold = split.folds[split_idx]

    # Define split indices.
    train_indices = fold.finetune + split.base_states + split.controls
    if use_validation_split:
        train_indices += fold.validation
    test_indices = fold.test

    # Initialize the baseline.
    if baseline_type not in BASELINES.keys():
        raise ValueError(f"Baseline {baseline_type} not supported. Pick one of: {list(BASELINES.keys())}")
    baseline = BASELINES[baseline_type]()

    # Load the ground truth dataset.
    ground_truth_paths = DatasetDirectory(root=ground_truth_path).model_dump()
    ground_truth = TxAnnotatedDataMatrix(**ground_truth_paths, var_gene_id_column=ground_truth_gene_id_column)

    # Subset the genes.
    # NOTE (cwognum): This assumes the ground truth genes are a superset of the prediction genes.
    #    This has been true so far, and an error will be raised in transform_single() if it is not.
    gene_subset = (
        pl.read_parquet(predictions_var_path)[predictions_gene_id_column]
        .unique(maintain_order=True)
        .to_list()
    )

    MatchGenesStep(gene_subset=gene_subset).transform_single(ground_truth)
    ScaleCountsStep(library_size=library_size).transform_single(ground_truth)
    Log1pStep().transform_single(ground_truth)

    type_adapter = TypeAdapter(TASK_ADAPTERS_TYPE)

    if task_id == "phenorescue":
        task_adapter = "drugscreen"
    elif task_id == "virtual_map":
        task_adapter = "singles"
    else:
        raise ValueError(f"Unknown task id: {task_id}")

    # Training

    # Since we eagerly load the ground truth, we need to deepcopy to be able to index it twice.
    # We might be able to do better here by implementing a custom copy method.
    logger.info(f"'Training' {baseline_type} baseline on {len(train_indices)} train indices.")

    ground_truth_train = deepcopy(ground_truth)
    ground_truth_train.filter(obs_indices=train_indices)
    training_task = type_adapter.validate_python({"kind": task_adapter, "dataset": ground_truth_train})
    training_task.prepare()
    baseline.fit(training_task)

    # Inference
    logger.info(f"Running inference on {len(test_indices)} test indices.")
    ground_truth_test = deepcopy(ground_truth)
    ground_truth_test.filter(obs_indices=test_indices)
    inference_task = type_adapter.validate_python({"kind": task_adapter, "dataset": ground_truth_test})
    inference_task.prepare()
    predictions = baseline.predict(inference_task)

    logger.info(f"Succesfully generated predictions. Shape: {predictions.shape}")

    # Write predictions to temporary directory, and call the evaluation CLI
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy the test obs from the ground truth.
        obs = pl.read_parquet(ground_truth.obs_path)
        obs = obs[test_indices]
        obs.write_parquet(tmpdir / "obs.parquet")

        # Save the predictions to a Zarr group.
        root = zarr.open(tmpdir / "features.zarr", mode="w")
        root.create_array("predictions", data=predictions)

        # Evaluate the predictions.
        logger.warning(
            "Running evaluation CLI. Be aware: We will redo some of the same transformations (e.g. gene matching) "
            "and you'll see logs related to this. This is not the most efficient, but prevents us having to "
            "either duplicate code between the evaluation and baseline CLI or permanently saving predictions to disk."
        )

        results = tx_evaluate_cli(
            predictions_path=tmpdir,
            ground_truth_path=ground_truth_path,
            task_id=task_id,
            split_path=split_path,
            split_idx=split_idx,
            save_destination=settings.save_dir,
            predictions_features_layer="predictions",
            predictions_zarr_index_column=None,
            predictions_var_path=predictions_var_path,
            library_size=library_size,
            use_validation_split=False,
            distributional_metrics=distributional_metrics,
            predictions_gene_id_column=predictions_gene_id_column,
            ground_truth_gene_id_column=ground_truth_gene_id_column,
            copy_base_states_and_controls=True,
        )

    return results
