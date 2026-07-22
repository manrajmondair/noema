#!/usr/bin/env bash
# Train Noema on one NLB dataset on a FarmShare L40S. Submit from the repo root
# after running scripts/farmshare_setup.sh:  sbatch scripts/farmshare_nlb.sbatch
# Pick a dataset with:  DATASET=mc_rtt sbatch scripts/farmshare_nlb.sbatch
# Warm-start from a pretrained backbone (fine-tune) by setting INIT:
#   sbatch --export=ALL,INIT=checkpoints/noema.pt scripts/farmshare_nlb.sbatch
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
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate noema
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p checkpoints

DATASET="${DATASET:-mc_maze}"
python -m noema.train.run \
  --dataset nlb --name "$DATASET" --path "data/$DATASET" \
  --dim "${DIM:-256}" --enc-depth "${ENC:-6}" --wm-depth "${WM:-3}" --heads "${HEADS:-8}" \
  --batch "${BATCH:-64}" --steps "${STEPS:-20000}" --lr "${LR:-3e-4}" --seed "${SEED:-0}" --ckpt "checkpoints/${SLURM_JOB_NAME:-noema}-${SLURM_JOB_ID}.pt" ${SPATIAL:+--spatial} ${CROSS:+--cross} ${POOL:+--attn-pool} ${CONTRAST:+--contrastive} ${SSM:+--ssm} ${HYBRID:+--hybrid} ${FILM:+--film} ${GRAFT:+--graft} ${SSM_STATE:+--ssm-state "$SSM_STATE"} ${SSM_DT:+--ssm-dt} ${W_CONTRAST:+--w-contrastive "$W_CONTRAST"} ${NMASK:+--neuron-mask "$NMASK"} ${INIT:+--init "$INIT"}
# co-bps and velocity R2 print to logs/noema-nlb-<jobid>.out at the end.
