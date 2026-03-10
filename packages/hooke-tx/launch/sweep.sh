#!/usr/bin/env bash

#SBATCH --array=1-2

#SBATCH --output=outputs/sweeps/out_%j_%a.out
#SBATCH --error=outputs/sweeps/error_%j_%a.out
#SBATCH --open-mode=append
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G

#SBATCH --partition=def
#SBATCH --qos=hooke-predict
#SBATCH --wckey=hooke-predict

set -e

cd /mnt/ps/home/CORP/$USER/projects/hooke-tx

export WANDB__SERVICE_WAIT=60

source .venv/bin/activate
module load CUDA/12.4.0

wandb agent --count 1 valencelabs/Hooke-Tx/[tbd:sweep_id]