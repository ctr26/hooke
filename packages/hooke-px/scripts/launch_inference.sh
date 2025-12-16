#!/bin/bash
# Launch distributed inference for multi-sample evaluation on multiple checkpoints
#
# Usage:
#   ./scripts/launch_inference.sh          # Launch all jobs in background
#   ./scripts/launch_inference.sh --wait   # Launch and wait for completion

# TODO: Update these paths for your specific checkpoint and parquet file
CHECKPOINT_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/1765902991/12149725/checkpoints"
OUTPUT_BASE="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/hooke-mini"
PARQUET="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/inference_set.parquet"
QOS="default"
# QOS="hooke-predict"

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

    OUTPUT_DIR="${OUTPUT_BASE}/step_${step}/"
    LOG_FILE="${OUTPUT_DIR}/master.log"

    mkdir -p "$OUTPUT_DIR"

    echo "Launching: step=${step} -> ${OUTPUT_DIR}"

    uv run run_distributed_inference.py \
        --setup.input_parquet "$PARQUET" \
        --setup.checkpoint "$CHECKPOINT" \
        --setup.output_dir "$OUTPUT_DIR" \
        --setup.num_workers 1000 \
        --setup.num_samples_per_image 36 \
        --setup.num_real_image_samples 1 \
        --setup.batch_size 8 \
        --setup.wandb_project big-img-eval \
        --setup.qos "$QOS" \
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
    echo "  tail -f ${OUTPUT_BASE}/step_*/master.log"
fi
