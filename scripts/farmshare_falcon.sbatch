#!/usr/bin/env bash
# FALCON H1 on a FarmShare L40S. Three modes, one script:
#
#   MODE=heldin   sbatch scripts/farmshare_falcon.sbatch   # official evaluator R2
#   MODE=fidelity sbatch scripts/farmshare_falcon.sbatch   # rollout firing correlation
#   MODE=sim2real sbatch scripts/farmshare_falcon.sbatch   # decoder-in-imagination
#
# Needs the noema-falcon env: falcon_challenge pulls a pandas that conflicts with the
# nlb_tools pin, so it lives in a clone rather than in the env the NLB path uses.
#
# The held-in and fidelity figures published earlier were measured before minival was
# found to be a byte-identical prefix of held-in-calib. Both entry points now excise
# the overlap through disjoint_calib, so runs from this script are not comparable to
# those numbers — that is the point of running it.
#SBATCH --job-name=noema-falcon
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate noema-falcon
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p checkpoints logs

MODE="${MODE:-heldin}"
COMMON=(--data "${DATA:-data/000954}" --task "${TASK:-h1}"
        --window "${WINDOW:-75}" --dim "${DIM:-256}" --batch "${BATCH:-64}"
        --steps "${STEPS:-6000}" --multistep "${MULTISTEP:-0}")

case "$MODE" in
  heldin)
    # Matches the configuration the withdrawn 0.87 was measured with, so the only
    # difference is the excision.
    python -m noema.eval.falcon_run "${COMMON[@]}" \
      --enc-depth "${ENC:-5}" --w-behavior "${W_BEHAVIOR:-6}"
    ;;
  fidelity|sim2real)
    python -m noema.eval.falcon_worldmodel "${COMMON[@]}" \
      --enc-depth "${ENC:-4}" --wm-depth "${WM:-3}" --seed "${SEED:-0}" \
      --ckpt "checkpoints/${SLURM_JOB_NAME:-falcon}-${SLURM_JOB_ID}.pt" \
      "--$MODE" ${HORIZONS:+--horizons "$HORIZONS"} ${STRIDE:+--stride "$STRIDE"}
    ;;
  *) echo "unknown MODE: $MODE" >&2; exit 2 ;;
esac
