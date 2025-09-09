import polars as pl
import typer
from tqdm import tqdm

from vcb.evaluate.dataloader import DrugscreenDataloader
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

app = typer.Typer()


@app.command()
def evaluate(
    predictions_path: str,
    ground_truth_path: str,
    results_path: str,
    load_to_memory: bool = False,
):
    """Evaluate a set of predictions against a ground truth."""

    predictions = Predictions(
        paths=PredictionsPaths(root=predictions_path), load_to_memory=load_to_memory
    )
    ground_truth = Dataset(
        paths=DatasetPaths(root=ground_truth_path), load_to_memory=load_to_memory
    )

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
