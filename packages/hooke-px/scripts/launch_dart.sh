#!/bin/bash
# Launch distributed inference for DART-style multi-sample evaluation on multiple checkpoints
#
# Usage:
#   ./scripts/launch_dart.sh          # Launch all jobs in background
#   ./scripts/launch_dart.sh --wait   # Launch and wait for completion

# TODO: Update these paths for your specific checkpoint and parquet file
CHECKPOINT_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/1765842211/12146024/checkpoints"
OUTPUT_BASE="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/big-img-dart"
PARQUET="/mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/overfit_data.parquet"

# Checkpoints to evaluate
STEPS=(50000)

WAIT_MODE=false
if [ "$1" == "--wait" ]; then
    WAIT_MODE=true
fi

cd /mnt/ps/home/CORP/jason.hartford/project/big-x/big-img

PIDS=()

for step in "${STEPS[@]}"; do
    CHECKPOINT="${CHECKPOINT_DIR}/step_${step}.ckpt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Checkpoint not found: $CHECKPOINT"
        continue
    fi

    OUTPUT_DIR="${OUTPUT_BASE}/step_${step}/dart"
    LOG_FILE="${OUTPUT_DIR}/master.log"

    mkdir -p "$OUTPUT_DIR"

    echo "Launching: step=${step} dataset=dart -> ${OUTPUT_DIR}"

    uv run run_distributed_inference.py \
        --setup.input_parquet "$PARQUET" \
        --setup.checkpoint "$CHECKPOINT" \
        --setup.output_dir "$OUTPUT_DIR" \
        --setup.num_workers 500 \
        --setup.num_samples_per_image 36 \
        --setup.num_real_image_samples 1 \
        --setup.batch_size 4 \
        --setup.wandb_project big-img-eval \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    echo "  PID: $! | Log: $LOG_FILE"
done

echo ""
echo "Launched ${#PIDS[@]} jobs"
echo "PIDs: ${PIDS[*]}"

if [ "$WAIT_MODE" == true ]; then
    echo ""
    echo "Waiting for all jobs to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
        echo "  PID $pid completed"
    done
    echo "All jobs complete!"
else
    echo ""
    echo "Jobs running in background. Monitor with:"
    echo "  tail -f ${OUTPUT_BASE}/step_*/*/master.log"
fi
