#!/bin/bash
# Launch distributed inference for ARPE19 evaluation
#
# Usage:
#   ./scripts/launch_inference-arpe19.sh          # Launch all jobs in background
#   ./scripts/launch_inference-arpe19.sh --wait   # Launch and wait for completion

TRAINING_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/1768305605/12583183"
OUTPUT_BASE="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/hooke-mini-v6/arpe19/XL"
PARQUET="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/arpe19_inference-valid_arpe19.parquet"

QOS="default"

# Checkpoints to evaluate
STEPS=(200000)

WAIT_MODE=false
if [ "$1" == "--wait" ]; then
    WAIT_MODE=true
fi

cd /mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/

PIDS=()

for step in "${STEPS[@]}"; do
    echo "Launching: step=${step}"

    python -m hooke_forge.inference.run \
        --training-dir "$TRAINING_DIR" \
        --step "$step" \
        --dataset "$PARQUET" \
        --output-base "$OUTPUT_BASE" \
        --num-workers 1000 \
        --num-samples 36 \
        --batch-size 3 \
        --qos "$QOS" \
        --build-maps \
        > "${OUTPUT_BASE}/step_${step}_master.log" 2>&1 &

    PIDS+=($!)
    echo "  PID: $! | Log: ${OUTPUT_BASE}/step_${step}_master.log"
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
    echo "  tail -f ${OUTPUT_BASE}/step_*_master.log"
fi
