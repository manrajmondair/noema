#!/usr/bin/env bash
# Score a rate-space ensemble on a GPU (fast). Members are globbed by PREFIX, which
# may list several colon-separated prefixes (comma collides with --export's separator)
# to pool architecturally distinct checkpoints into one diverse ensemble.
#   sbatch --export=ALL,PREFIX=noema-honest:noema-cross-s,DATASET=mc_maze scripts/farmshare_ensemble.sbatch
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
PREFIXES="${PREFIX:-noema}"
CKPTS=$(for p in ${PREFIXES//:/ }; do ls checkpoints/${p}*.pt 2>/dev/null; done | sort -u | paste -sd,)
python -m noema.eval.ensemble_run --ckpts "$CKPTS" --name "$DATASET" --path "data/$DATASET"
