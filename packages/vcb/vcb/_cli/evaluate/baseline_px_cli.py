from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.split import Split
from vcb.baselines.baselines import (
    ContextMeanBaseline,
    PerturbationMeanBaseline,
    ContextSampleBaseline,
    PerturbationSampleBaseline,
)
from vcb.models.baseline_predictions import InMemoryPredictions
from vcb.evaluate.evaluate import evaluate
import numpy as np
from tqdm import tqdm


baseline_lookup = {
    "context_mean": ContextMeanBaseline,
    "context_sample": ContextSampleBaseline,
    "perturbation_mean": PerturbationMeanBaseline,
    "perturbation_sample": PerturbationSampleBaseline,
}


def run_baseline_px_cli(
    root: str,
    split_path: str,
    split_idx: int,
    baseline_type: str,
    results_path: str,
):
    """
    Evaluate baseline performance on a phenomics embedding dataset.

    Args:
        root: Path to the root directory of the dataset.
        split_path: Path to the split json file.
        split_idx: Index of the split to evaluate.
        baseline_type: Type of baseline to evaluate.
            Options: context_mean, context_sample, perturbation_mean, perturbation_sample
        results_path: Path to the results parquet file.
    """

    ground_truth = Dataset.from_directory(DatasetDirectory(root=root))
    split = Split.from_json(split_path)

    fold = split.folds[split_idx]

    # define split indices
    finetune_split = fold.finetune + split.controls
    test_split = fold.test + split.controls

    # init and cache baseline
    if baseline_type not in baseline_lookup.keys():
        raise ValueError(f"Baseline {baseline_type} not supported")
    else:
        BaselineType = baseline_lookup[baseline_type]
        baseline = BaselineType(ground_truth, valid_indices=finetune_split)

    # get baseline predictions
    # TODO: add variable number sample generation for N2D models
    predictions_list = []
    for obs_row in tqdm(ground_truth.obs[test_split].iter_rows(named=True)):
        output = baseline.forward(obs_row)
        predictions_list.append(output)

    # create datasets for evaluation
    predictions = InMemoryPredictions(
        obs=ground_truth.obs[test_split], X=np.stack(predictions_list)
    )

    # Evaluate and save the results
    results = evaluate(predictions, ground_truth)
    results.write_parquet(results_path)
