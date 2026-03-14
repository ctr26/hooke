#!/bin/bash
# Launch distributed brightfield embedding inference
#
# Usage:
#   ./scripts/launch_bf_embedding.sh                    # Launch job in background (incomplete rows only)
#   ./scripts/launch_bf_embedding.sh --wait             # Launch and wait for completion (incomplete rows only)
#   ./scripts/launch_bf_embedding.sh --full             # Launch full reprocess in background (all rows)
#   ./scripts/launch_bf_embedding.sh --full --wait      # Launch full reprocess and wait (all rows)

# Dataset configuration
DATASET_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/hooke-mini-v6/cross_cell_line/XL/step_300000"
INPUT_METADATA="${DATASET_DIR}/prepared_metadata.parquet"
INPUT_IMAGES="${DATASET_DIR}/features/pred_images.zarr"
OUTPUT_DIR="$DATASET_DIR"  # Same directory - will create pred_phenom_BF.zarr alongside existing files

# Job configuration
NUM_WORKERS=300
BATCH_SIZE=4
PARTITION="hopper"
QOS="default"

# Parse command line arguments
WAIT_MODE=false
FORCE_REPROCESS=false

for arg in "$@"; do
    case $arg in
        --wait)
            WAIT_MODE=true
            ;;
        --full)
            FORCE_REPROCESS=true
            ;;
    esac
done

# Change to big-img directory
cd /rxrx/data/user/jason.hartford/claude/big-img

# Validate input files
if [ ! -f "$INPUT_METADATA" ]; then
    echo "Error: Input metadata not found: $INPUT_METADATA"
    exit 1
fi

if [ ! -d "$INPUT_IMAGES" ]; then
    echo "Error: Input images not found: $INPUT_IMAGES"
    exit 1
fi

echo "=== Brightfield Embedding Inference Configuration ==="
echo "Input metadata: $INPUT_METADATA"
echo "Input images: $INPUT_IMAGES"
echo "Output directory: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Partition: $PARTITION"
if [ "$FORCE_REPROCESS" == true ]; then
    echo "Mode: FULL REPROCESS (all 118,717 rows)"
else
    echo "Mode: INCREMENTAL (incomplete rows only)"
fi
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set up log file
LOG_FILE="${OUTPUT_DIR}/bf_embedding_master.log"

echo "Launching brightfield embedding inference..."
echo "Log file: $LOG_FILE"

# Build base command
BASE_CMD="uv run run_bf_embedding_inference.py \
    --setup.input_metadata \"$INPUT_METADATA\" \
    --setup.input_images \"$INPUT_IMAGES\" \
    --setup.output_dir \"$OUTPUT_DIR\" \
    --setup.num_workers \"$NUM_WORKERS\" \
    --setup.batch_size \"$BATCH_SIZE\" \
    --setup.partition \"$PARTITION\" \
    --setup.qos \"$QOS\""

# Add force_reprocess flag if requested
if [ "$FORCE_REPROCESS" == true ]; then
    BASE_CMD="$BASE_CMD --setup.force_reprocess true"
fi

if [ "$WAIT_MODE" == true ]; then
    # Run in foreground and wait
    eval $BASE_CMD
else
    # Run in background
    eval $BASE_CMD > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "Job launched in background with PID: $PID"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check SLURM jobs with:"
    echo "  squeue -u $(whoami)"
fi