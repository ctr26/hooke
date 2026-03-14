from vcb.baselines.mean.delta import MeanContextDeltaBaseline, MeanPerturbationDeltaBaseline
from vcb.baselines.mean.sample import MeanContextSampleBaseline, MeanPerturbationSampleBaseline

BASELINES = {
    "mean_context_delta": MeanContextDeltaBaseline,
    "mean_perturbation_delta": MeanPerturbationDeltaBaseline,
    "mean_context_sample": MeanContextSampleBaseline,
    "mean_perturbation_sample": MeanPerturbationSampleBaseline,
}
