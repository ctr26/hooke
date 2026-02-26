#!/bin/bash
#SBATCH --job-name=hooke-gene-select
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/%j/gene_select.log

set -euo pipefail

cd /rxrx/data/user/jason.hartford/big-x/big-img
mkdir -p outputs/${SLURM_JOB_ID}

echo "Starting gene selection job ${SLURM_JOB_ID}"

OUTPUT_PATH="outputs/${SLURM_JOB_ID}/gene_subset_hvg_5000.npz"

.venv/bin/python3 .venv/bin/hooke-gene-select \
    --gene_selection.metadata_path /mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata/training_trek_v1_0_obs.parquet \
    --gene_selection.select_strategy hvg \
    --gene_selection.n_features 5000 \
    --gene_selection.output_path ${OUTPUT_PATH}

echo "Gene selection complete. Output at: ${OUTPUT_PATH}"
echo "GENE_SUBSET_PATH=${OUTPUT_PATH}"
