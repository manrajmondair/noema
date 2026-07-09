#!/usr/bin/env bash
# Fetch a Neural Latents Benchmark dataset and train Noema on it.
# Usage: scripts/nlb.sh [dataset] [dandiset]   (defaults: mc_maze 000128)
set -euo pipefail

DATASET="${1:-mc_maze}"
DANDISET="${2:-000128}"
DATA_DIR="data/${DATASET}"

pip install -e ".[data,train]"

if [ ! -d "$DATA_DIR" ]; then
  mkdir -p data
  dandi download "DANDI:${DANDISET}/draft" -o data/
  # dandi lays the dataset out under data/<dandiset>/; point the loader at it
  mv "data/${DANDISET}" "$DATA_DIR"
fi

python -m noema.train.run \
  --dataset nlb --name "$DATASET" --path "$DATA_DIR" \
  --dim 256 --enc-depth 6 --wm-depth 3 --heads 8 \
  --batch 64 --steps 20000 --lr 3e-4 --wandb
