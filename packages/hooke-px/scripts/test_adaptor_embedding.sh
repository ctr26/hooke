#!/bin/bash
#SBATCH --job-name=test-adaptor-emb
#SBATCH --partition=hopper
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/%j/test_adaptor_embedding.log

# Test adapter embedding extraction with a small subset
#
# This script runs a quick test to validate the adaptor embedding extraction
# before launching a full distributed job.
#
# Usage:
#   sbatch scripts/test_adaptor_embedding.sh [case]
#   # case: 1, 2, or 3 (default: 3, which is the smallest)

set -e

# Create output directory for logs
mkdir -p outputs/${SLURM_JOB_ID}

CASE="${1:-3}"
CHECKPOINT="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/1765615249/12128283/checkpoints/step_50000_with_tokenizer.ckpt"
PARQUET="/mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/pretraining_v2.parquet"
OUTPUT_BASE="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/adaptor_embedding_test"

cd /mnt/ps/home/CORP/jason.hartford/project/big-x/big-img

echo "========================================"
echo "Testing Adapter Embedding Extraction"
echo "========================================"
echo "Case: $CASE"
echo "Checkpoint: $CHECKPOINT"
echo "Parquet: $PARQUET"
echo ""

# Check files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$PARQUET" ]; then
    echo "ERROR: Parquet not found: $PARQUET"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/case_${CASE}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with --skip_launch to just prepare the data and zarr
echo "Step 1: Preparing metadata (skip_launch mode)..."
uv run run_distributed_adaptor_embedding.py \
    --setup.input_parquet "$PARQUET" \
    --setup.checkpoint "$CHECKPOINT" \
    --setup.output_dir "$OUTPUT_DIR" \
    --setup.case "$CASE" \
    --setup.num_workers 1 \
    --setup.batch_size 64 \
    --setup.skip_launch True

echo ""
echo "Step 2: Running worker locally on first chunk..."

# Run the worker directly
if [ -d "${OUTPUT_DIR}/workers/worker_0" ]; then
    uv run run_adaptor_embedding.py \
        --worker_dir "${OUTPUT_DIR}/workers/worker_0" \
        --config "${OUTPUT_DIR}/config.json"
else
    echo "No worker directory created (may be all complete already)"
fi

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Embeddings:"
ls -la "${OUTPUT_DIR}/embeddings/" 2>/dev/null || echo "No embeddings directory yet"
echo ""
echo "To inspect the zarr:"
echo "  python -c \"import zarr; z = zarr.open('${OUTPUT_DIR}/embeddings/adaptor_emb.zarr'); print(f'Shape: {z.shape}, dtype: {z.dtype}')\""
