from vcb.data_models.metrics.base import Metric, MetricInfo
from vcb.metrics.distributional.mmd import compute_e_distance
from vcb.metrics.retrieval import calculate_edistance_retrieval, calculate_mae_retrieval
from vcb.metrics.simple import cosine, cosine_delta, mse, pearson, pearson_delta

METRICS: dict[Metric, MetricInfo] = {
    "edistance": MetricInfo(fn=compute_e_distance, is_distributional=True),
    "mse": MetricInfo(fn=mse),
    "pearson": MetricInfo(fn=pearson),
    "cosine": MetricInfo(fn=cosine),
    "pearson_delta": MetricInfo(fn=pearson_delta, is_delta_metric=True),
    "cosine_delta": MetricInfo(fn=cosine_delta, is_delta_metric=True),
    "retrieval_mae": MetricInfo(
        fn=calculate_mae_retrieval, is_delta_metric=False, kwargs={"use_deltas": False}
    ),
    "retrieval_mae_delta": MetricInfo(
        fn=calculate_mae_retrieval, is_delta_metric=True, kwargs={"use_deltas": True}
    ),
    "retrieval_edistance": MetricInfo(
        fn=calculate_edistance_retrieval,
        is_distributional=True,
        is_delta_metric=True,
    ),
}
