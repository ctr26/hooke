from vcb.data_models.config import EvaluationConfig
from vcb.data_models.metrics.suites.pep import PerturbationEffectPredictionSuite
from vcb.data_models.metrics.suites.retrieval import RetrievalSuite
from vcb.data_models.task.drugscreen import DrugscreenTaskAdapter
from vcb.preprocessing.pipeline import TranscriptomicsPreprocessingPipeline
from vcb.preprocessing.steps.match_genes import MatchGenesStep


def test_config_serialization(
    tmpdir, mock_drugscreen_dataset, mock_drugscreen_predictions, mock_drugscreen_split_path
):
    config = EvaluationConfig(
        ground_truth=DrugscreenTaskAdapter(dataset=mock_drugscreen_dataset),
        predictions=DrugscreenTaskAdapter(dataset=mock_drugscreen_predictions),
        split_path=mock_drugscreen_split_path,
        split_index=0,
        use_validation_split=False,
        preprocessing_pipeline=TranscriptomicsPreprocessingPipeline(
            steps=[
                MatchGenesStep(
                    ground_truth_gene_id_column="ensembl_gene_id",
                    predictions_gene_id_column="ensembl_gene_id",
                ),
            ]
        ),
        metric_suites=[
            RetrievalSuite(
                metric_labels={"retrieval_mae", "retrieval_edistance"},
                use_distributional_metrics=True,
            ),
            PerturbationEffectPredictionSuite(
                metric_labels={"cosine", "cosine_delta", "mse", "pearson", "pearson_delta"},
                use_distributional_metrics=False,
            ),
        ],
    )

    with open(tmpdir / "config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))

    with open(tmpdir / "config.json", "r") as f:
        deserialized = EvaluationConfig.model_validate_json(f.read())

    # Using the model_dump() (note: Not model_dump_json()) here is a lazy way to compare the data models.
    # Since I don't need to worry about comparing the DataFrames and other complex objects inside the data models.
    expected = config.model_dump(exclude={"timestamp"})
    actual = deserialized.model_dump(exclude={"timestamp"})
    assert expected == actual
