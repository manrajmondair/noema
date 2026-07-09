#!/usr/bin/env bash
# Stage 1: self-supervised multi-session pretraining across several NLB datasets.
# Each dataset's neurons occupy a disjoint slice of a shared embedding table and
# carry a session label for the adversarial invariance term.
set -euo pipefail

fetch() {  # name dandiset
  if [ ! -d "data/$1" ]; then
    dandi download "DANDI:$2/draft" -o data/
    mv "data/$2" "data/$1"
  fi
}

pip install -e ".[data,train]"
fetch mc_maze 000128
fetch mc_rtt 000129
fetch area2_bump 000127

python -m noema.train.run \
  --datasets mc_maze,mc_rtt,area2_bump --data-root data \
  --dim 256 --enc-depth 6 --wm-depth 3 --heads 8 \
  --batch 64 --steps 40000 --lr 3e-4 --wandb

# Stage 2: fine-tune on one dataset with its behavior labels, warm-starting the
# pretrained backbone (fresh behavior head):
#   python -m noema.train.run --dataset nlb --name mc_maze --path data/mc_maze \
#     --init checkpoints/noema.pt --steps 5000 --wandb
