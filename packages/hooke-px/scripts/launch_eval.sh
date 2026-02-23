#!/bin/bash
# Launch distributed inference for evaluation on multiple checkpoints and datasets
#
# Usage:
#   ./scripts/launch_eval.sh          # Launch all jobs in background
#   ./scripts/launch_eval.sh --wait   # Launch and wait for completion

# TODO: Update these paths for your specific checkpoint and parquet file
CHECKPOINT_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/outputs/YOUR_JOB_ID/checkpoints"
OUTPUT_BASE="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/big-img-eval"
PARQUET="/path/to/your/evaluation_data.parquet"

# Checkpoints to evaluate
STEPS=(50000 100000 150000 200000)

# Datasets to evaluate (split, source, name, num_workers)
# Format: "split:source:name:num_workers"
DATASETS=(
    "val:iid:val_iid:50"
    "test:iid:test_iid:50"
)

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

    for dataset_info in "${DATASETS[@]}"; do
        IFS=':' read -r split source name num_workers <<< "$dataset_info"

        OUTPUT_DIR="${OUTPUT_BASE}/step_${step}/${name}"
        LOG_FILE="${OUTPUT_DIR}/master.log"

        mkdir -p "$OUTPUT_DIR"

        echo "Launching: step=${step} dataset=${name} -> ${OUTPUT_DIR}"

        uv run -m hooke_forge.inference.run_distributed \
            --setup.input_parquet "$PARQUET" \
            --setup.checkpoint "$CHECKPOINT" \
            --setup.output_dir "$OUTPUT_DIR" \
            --setup.split "$split" \
            --setup.source "$source" \
            --setup.num_workers "$num_workers" \
            --setup.num_samples_per_image 1 \
            --setup.batch_size 8 \
            --setup.wandb_project big-img-eval \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        echo "  PID: $! | Log: $LOG_FILE"
    done
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
