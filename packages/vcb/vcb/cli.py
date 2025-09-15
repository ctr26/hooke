import polars as pl
import typer
from loguru import logger
from tqdm import tqdm

from vcb.data.dataloader import DrugscreenDataloader
from vcb.data.raw_count_scaler import RawCountScaler
from vcb.evaluate.evaluate import (
    calculate_aggregated_metrics,
    calculate_distributional_metrics,
)
from vcb.evaluate.match import yield_batch_pairs, yield_compound_pairs
from vcb.evaluate.utils import add_compound_perturbation_to_obs
from vcb.metrics.retrieval import calculate_edistance_retrieval, calculate_mae_retrieval
from vcb.models.dataset import Dataset, DatasetPaths
from vcb.models.predictions import Predictions, PredictionsPaths
from vcb.models.split import Split

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def evaluate(
    predictions_path: str,
    ground_truth_path: str,
    results_path: str,
    predictions_var_path: str | None = None,
    predictions_features_array_name: str | None = None,
    predictions_gene_id_column: str | None = "ensembl_gene_id",
    ground_truth_gene_id_column: str | None = "ensembl_gene_id",
    match_gene_labels: bool = False,
):
    """
    Evaluate a set of predictions against a ground truth.

    TODO (cwognum): The --match-gene-labels flag is a temporary, manual solution.
        Ideally, we would specify through the CLI what the modality is, or maybe even use entirely separate CLI commands.
        We'll for example also need to specify whether we're using raw counts, embeddings, etc. We're going to run into
        this relatively soon (once we want to evaluate phenomics), but I'm going to punt on this for now.
    """

    predictions = Predictions(
        paths=PredictionsPaths(
            root=predictions_path,
            var_path=predictions_var_path,
        ),
        features_array_name=predictions_features_array_name,
        gene_id_column=predictions_gene_id_column,
    )

    ground_truth = Dataset(
        paths=DatasetPaths(root=ground_truth_path),
        gene_id_column=ground_truth_gene_id_column,
    )

    if match_gene_labels:
        intersection = set(predictions.gene_labels) & set(ground_truth.gene_labels)
        predictions.set_gene_labels_subset(intersection)
        ground_truth.set_gene_labels_subset(intersection)

    scaler = RawCountScaler()
    scaler.fit(ground_truth.X)
    predictions.X = scaler.transform(predictions.X, is_log1p_transformed=True)
    ground_truth.X = scaler.transform(ground_truth.X)

    rows = []

    def extend_rows(scores, batch_definition):
        """Helper function to extend the rows list with the scores and batch definition."""
        for k, v in scores.items():
            if isinstance(v, list):
                for i, v_i in enumerate(v):
                    rows.append(
                        {
                            **batch_definition,
                            "score": v_i,
                            "metric": k,
                            "sample_index": i,
                        }
                    )
            else:
                rows.append({**batch_definition, "score": v, "metric": k})
        return rows

    logger.info("Calculating batch-level metrics.")
    for pred, truth, base, batch_definition, compounds_pred, compounds_truth in tqdm(
        yield_batch_pairs(predictions, ground_truth),
        total=predictions.obs["batch_center"].n_unique(),
    ):
        scores = calculate_edistance_retrieval(
            samples_pred=pred,
            samples_truth=truth,
            group_labels_pred=compounds_pred,
            group_labels_truth=compounds_truth,
        )
        extend_rows(scores, batch_definition)

        scores = calculate_mae_retrieval(
            samples_pred=pred,
            samples_truth=truth,
            group_labels_pred=compounds_pred,
            group_labels_truth=compounds_truth,
        )
        extend_rows(scores, batch_definition)

    # This is not strictly needed, but makes for nicer progress bars.
    logger.info("Calculating compound-level metrics.")
    query = add_compound_perturbation_to_obs(
        predictions.obs.filter(pl.col("drugscreen_query")).filter(
            pl.col("perturbations").list.len() == 2
        )
    )
    total = query[
        "inchikey",
        "concentration",
        "batch_center",
        *ground_truth.metadata.biological_context,
    ].n_unique()

    for pred, truth, base, batch_definition in tqdm(
        yield_compound_pairs(predictions, ground_truth),
        total=total,
    ):
        scores = calculate_aggregated_metrics(pred, truth, base)
        extend_rows(scores, batch_definition)

        scores = calculate_distributional_metrics(pred, truth)
        extend_rows(scores, batch_definition)

    pl.DataFrame(rows).write_parquet(results_path)


@app.command()
def finetune(root: str, split_path: str):
    # NOTE (cwognum): I'm using this CLI as a scrappy test bed. Don't judge me.

    dataset = Dataset(paths=DatasetPaths(root=root))
    split = Split.from_json(split_path)

    for fold in split.folds:
        # Finetune your model here...
        dataloader = DrugscreenDataloader(dataset=dataset, indices=fold.finetune)
        (control, base, perturbed, perturbations, biological_context) = dataloader[0]
        print(
            control.shape,
            base.shape,
            perturbed.shape,
            perturbations,
            biological_context,
        )

        # Evaluate your model here...
        dataloader = DrugscreenDataloader(dataset=dataset, indices=fold.test)
        (control, base, perturbed, perturbations, biological_context) = dataloader[0]
        print(
            control.shape,
            base.shape,
            perturbed.shape,
            perturbations,
            biological_context,
        )

        break


if __name__ == "__main__":
    app()
