#!/bin/bash
#SBATCH --job-name=prepare-hf-cache
#SBATCH --partition=cpu
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/%j/prepare_hf_cache.log

cd /mnt/ps/home/CORP/jason.hartford/project/big-x/big-img
mkdir -p outputs/${SLURM_JOB_ID}

uv run scripts/prepare_hf_cache.py \
    --path /mnt/ps/home/CORP/jason.hartford/project/big-x/joint-model/metadata/pretraining_v2.parquet \
    --chunk-size 100000