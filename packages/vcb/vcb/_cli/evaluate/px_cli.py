from vcb.evaluate.evaluate import evaluate
from vcb.models.anndata import AnnotatedDataMatrix
from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.predictions import PredictionPaths


def px_evaluate_cli(
    predictions_path: str,
    ground_truth_path: str,
    results_path: str,
    predictions_features_layer: str,
    predictions_var_path: str | None = None,
):
    """
    Evaluate predictions in Phenomics against a ground truth.

    Args:
        predictions_path: Path to the predictions directory.
        ground_truth_path: Path to the ground truth directory.
        results_path: Path to the results parquet file.
        predictions_features_layer: Layer of the features to use for the predictions.
        predictions_var_path: (optional) Path to the var file for the predictions.
    """

    # Load the predictions.
    predictions = AnnotatedDataMatrix(
        **PredictionPaths(
            root=predictions_path,
            var_path=predictions_var_path,
        ).model_dump(),
        features_layer=predictions_features_layer,
    )

    # Load the ground truth.
    ground_truth = Dataset.from_directory(DatasetDirectory(root=ground_truth_path))

    # Evaluate and save the results
    results = evaluate(predictions, ground_truth)
    results.write_parquet(results_path)
