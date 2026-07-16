#!/usr/bin/env bash
# Self-supervised cross-subject pretraining across NLB datasets on a FarmShare
# L40S, saving checkpoints/noema.pt. Run setup for all three datasets first:
#   scripts/farmshare_setup.sh mc_maze mc_rtt area2_bump
#   sbatch scripts/farmshare_pretrain.sbatch
# Then fine-tune one dataset from the pretrained backbone:
#   DATASET=mc_maze sbatch --export=ALL,INIT=checkpoints/noema.pt scripts/farmshare_nlb.sbatch
#SBATCH --job-name=noema-pretrain
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate noema
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p checkpoints

python -m noema.train.run \
  --datasets mc_maze,mc_rtt,area2_bump --data-root data \
  --dim 256 --enc-depth 6 --wm-depth 3 --heads 8 \
  --batch 64 --steps 40000 --lr 3e-4 --seed 0 --ckpt checkpoints/pretrain.pt
