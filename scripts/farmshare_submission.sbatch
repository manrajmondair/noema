#!/usr/bin/env bash
# Build the NLB submission .h5 from a diverse ensemble. PREFIX may list several
# colon-separated checkpoint prefixes, as in farmshare_ensemble.sbatch.
#   sbatch --export=ALL,PREFIX=noema-honest:noema-cross-s,DATASET=mc_maze,MINT=1 scripts/farmshare_submission.sbatch
# MINT=1 appends the non-NN trajectory-library member; FORWARD=1 writes forward-prediction keys.
#SBATCH --job-name=noema-submission
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
python -m noema.eval.submission --ckpts "$CKPTS" --name "$DATASET" --path "data/$DATASET" --out "${OUT:-submission.h5}" ${MINT:+--mint} ${FORWARD:+--forward} ${TEACHER:+--teacher}
