from pathlib import Path

import polars as pl
from loguru import logger

from vcb.data_models.config import EvaluationConfig
from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.data_models.dataset.dataset_directory import DatasetDirectory
from vcb.data_models.dataset.predictions import PredictionPaths
from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.split import Split
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.preprocessing.match_genes import match_gene_space
from vcb.preprocessing.scale_counts import TxDistributionHandler 


def tx_evaluate_cli(
    predictions_path: Path,
    ground_truth_path: Path,
    split_path: Path,
    split_idx: int,
    save_destination: Path,
    predictions_features_layer: str,
    predictions_zarr_index_column: str,
    predictions_var_path: Path,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
    library_size: int | None = None,
    distributional_metrics: bool = True,
    log1p_predictions: bool = False,
    rescale_predictions: bool = True,
    use_val_split: bool = False,
):
    """
    Evaluate predictions in Transcriptomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory.
        ground_truth_path: Path to the ground truth directory.
        split_path: Path to the split json file.
        split_idx: Index of the split to evaluate.
        save_destination: Path to where results should be saved.
        predictions_var_path: Path to the var file for the predictions.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_zarr_index_column: Column of the predictions to use for the zarr index.
        predictions_gene_id_column: (optional) Column of the predictions to use for the gene id.
        ground_truth_gene_id_column: (optional) Column of the ground truth to use for the gene id.
        library_size: (optional) Library size to use for the evaluation (default ground truth median library size).
        distributional_metrics: (optional) Whether to include distributional metrics.
        log1p_predictions: (optional) Log1p the predictions (default False, assuming this is done).
        rescale_predictions: (optional) Rescale the predictions to a target library size (default True).
        use_val_split: (optional) Whether to use the validation split instead of the test split (default False).
    NOTE (cwognum): For now, this only supports the count space. We don't yet support evaluation in embedding spaces.
    """

    # Load the ground truth.
    ground_truth = AnnotatedDataMatrix(**DatasetDirectory(root=ground_truth_path).model_dump())

    # Load the split to filter down the ground truth.
    split = Split.from_json(split_path)
    fold = split.folds[split_idx]
    if use_val_split:
        split_indices = fold.validation + split.base_states
    else:
        split_indices = fold.test + split.base_states
    ground_truth.set_obs_indices(split_indices)

    # Load the predictions.
    predictions = AnnotatedDataMatrix(
        **PredictionPaths(root=predictions_path).model_dump(),
        var_path=predictions_var_path,
        metadata_path=ground_truth.metadata_path,
        features_layer=predictions_features_layer,
        zarr_index_column=predictions_zarr_index_column,
    )

    # Match the gene space
    predictions, ground_truth = match_gene_space(
        predictions,
        ground_truth,
        predictions_gene_id_column,
        ground_truth_gene_id_column,
    )

    # Scale to a consistent library size
    tx_scaling_handler = TxDistributionHandler(ground_truth, predictions, library_size, 
                                               rescale_predictions=rescale_predictions, 
                                               log1p_predictions=log1p_predictions)
    
    tx_scaling_handler.scale_both_as_needed()

    config = EvaluationConfig(
        metric_suites=[
            RetrievalSuite(
                ground_truth=DrugscreenTaskAdapter(
                    dataset=ground_truth,
                    context_groupby_cols={*ground_truth.metadata.biological_context, "plate_disease_model"},
                ),
                predictions=DrugscreenTaskAdapter(
                    dataset=predictions,
                    context_groupby_cols={*predictions.metadata.biological_context, "plate_disease_model"},
                ),
                metric_labels={"retrieval_mae", "retrieval_mae_delta", "retrieval_edistance"},
                use_distributional_metrics=distributional_metrics,
            ),
            PerturbationEffectPredictionSuite(
                ground_truth=DrugscreenTaskAdapter(
                    dataset=ground_truth,
                    context_groupby_cols={
                        *ground_truth.metadata.biological_context,
                        "batch_center",
                        "plate_disease_model",
                    },
                ),
                predictions=DrugscreenTaskAdapter(
                    dataset=predictions,
                    context_groupby_cols={
                        *predictions.metadata.biological_context,
                        "batch_center",
                        "plate_disease_model",
                    },
                ),
                metric_labels={"pearson", "pearson_delta", "cosine", "cosine_delta", "mse"},
                use_distributional_metrics=distributional_metrics,
            ),
        ],
    )

    # Evaluate
    results = config.execute()

    # Save the results
    save_destination.mkdir(parents=True, exist_ok=True)
    results.write_parquet(save_destination / "results.parquet")
    with open(save_destination / "config.json", "w") as f:
        # TODO (cwognum): This is not a perfect serialization, because we don't persist which dataset subclass was used.
        f.write(config.model_dump_json(indent=4))

    # Summarize the results
    summary = (
        results.group_by("metric")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
        .sort("metric")
    )
    print(summary)
