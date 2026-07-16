#!/usr/bin/env bash
# Score a rate-space ensemble on a GPU (fast). Pass the checkpoint list via CKPTS:
#   CKPTS=$(ls checkpoints/noema-*.pt | paste -sd,) sbatch --export=ALL,CKPTS="$CKPTS" scripts/farmshare_ensemble.sbatch
#SBATCH --job-name=noema-ens-eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate noema

DATASET="${DATASET:-mc_maze}"
python -m noema.eval.ensemble_run --ckpts "$CKPTS" --name "$DATASET" --path "data/$DATASET"
