#!/usr/bin/env bash
# Train Noema on one NLB dataset on a FarmShare L40S. Submit from the repo root
# after running scripts/farmshare_setup.sh:  sbatch scripts/farmshare_nlb.sbatch
# Pick a dataset with:  DATASET=mc_rtt sbatch scripts/farmshare_nlb.sbatch
#SBATCH --job-name=noema-nlb
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DATASET="${DATASET:-mc_maze}"
python -m noema.train.run \
  --dataset nlb --name "$DATASET" --path "data/$DATASET" \
  --dim 256 --enc-depth 6 --wm-depth 3 --heads 8 \
  --batch 64 --steps 20000 --lr 3e-4 --seed 0
# co-bps and velocity R2 print to logs/noema-nlb-<jobid>.out at the end.
