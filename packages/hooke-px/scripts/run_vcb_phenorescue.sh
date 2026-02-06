#!/bin/bash
#SBATCH --job-name=run-vcb-phenorescue
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --output=outputs/run_vcb_phenorescue_%j.log

PREDICTIONS_PATH=/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/hooke-mini-v6/drugscreen/XL/step_200000 
GROUND_TRUTH_PATH=/rxrx/data/valence/internal_benchmarking/vcds1/drugscreen__cell_paint__v1_2
SPLIT_PATH=/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__cell_paint__v1_2/split_compound_random__v1.json
OUTPUT_PATH="${PREDICTIONS_PATH}_eval"

echo "========================================"
echo "Running VCB Phenorescue Evaluation"
echo "========================================"
echo "Predictions Path: $PREDICTIONS_PATH"
echo "Ground Truth Path: $GROUND_TRUTH_PATH"
echo "Split Path: $SPLIT_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "Current time: $(date)"
echo "Current time in London: $(TZ=':Europe/London' date)"
echo ""
cd /mnt/ps/home/CORP/jason.hartford/project/vcb_main
uv run vcb evaluate predictions px -p $PREDICTIONS_PATH -t $GROUND_TRUTH_PATH -s $SPLIT_PATH -o $OUTPUT_PATH --task-id phenorescue --predictions-zarr-index-column zarr_index --no-distributional-metrics --no-copy-base-states-and-controls
