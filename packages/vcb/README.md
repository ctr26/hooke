# Virtual Cell Benchmark (VCB)

Intended to be a temporary repository, but you never know...

## Install

We recommend using [`uv`](https://docs.astral.sh/uv/).
```bash
uv sync
```

## Usage

Installing this package makes a CLI command available.

### Evaluate model predictions:
```bash
uv run vcb predictions --help
uv run vcb predictions [tx/px/...] <path_to_predictions_dir> <path_to_groundtruth_dir> <results_save_path> <feature_layer_name> <path_to_predictions_var (tx only)>
```

### Evaluate baselines:
```bash
uv run vcb baseline --help
uv run vcb baseline [tx/px/...] <path_to_groundtruth_dir> <path_to_split_json> <split_index> <baseline_type> <results_save_path>
```

<!-- ### Draft command for active dev on baseline

Hooke-1 prediction evaluation:
```
uv run vcb predictions tx "/rxrx/data/user/andrew.quirke/outgoing/drugscreen_tx_benchmark" "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__trekseq__v1_0" "./test.pq" /rxrx/data/user/cas.wognum/outgoing/txfm_v0_2g2zvul8__gene_labels.parquet tx_raw_counts --predictions-gene-id-column "gene_label"
```

```
uv run vcb predictions px "/rxrx/data/user/andrew.quirke/outgoing/phx_inf2/drugscreen_phx_benchmark/fold_o0_i0_r0/o0_i0__20250912_093245/" "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_0" "./test.pq" ph2_embeddings
```

Trekseq baselines:
```
uv run vcb baseline tx "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__trekseq__v1_0" "/rxrx/data/user/cas.wognum/outgoing/vcb/2025_09_vcb_benchmark_v1/drugscreen__trekseq__v1_0__split_random__v1_0.json" 0 "context_mean" "./baseline_predictions.parquet"
```

CellPaint baselines:
```
uv run vcb baseline px "/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_0" "/rxrx/data/user/cas.wognum/outgoing/vcb/2025_09_vcb_benchmark_v1/drugscreen__cell_paint__v1_0__split_random__v1_0.json" 0 "context_mean" "./baseline_predictions.parquet"
``` -->