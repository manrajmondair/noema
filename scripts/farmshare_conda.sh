#!/usr/bin/env bash
# Environment setup on a FarmShare LOGIN node via Miniforge.
#
# nlb_tools requires pandas <=1.3.4, which has no wheels for FarmShare's system
# Python (3.12+), and FarmShare offers neither an older Python module nor conda.
# So we bring our own: a Python 3.10 conda env with the nlb_tools-compatible
# scientific stack, plus CUDA torch. Also downloads any datasets named as args
# (default mc_maze) while the login node has internet.
#   scripts/farmshare_conda.sh                            # env + mc_maze
#   scripts/farmshare_conda.sh mc_maze mc_rtt area2_bump  # + pretraining datasets
set -euo pipefail
cd "$(dirname "$0")/.."

dandiset() { case "$1" in
  mc_maze) echo 000128 ;; mc_rtt) echo 000129 ;;
  area2_bump) echo 000127 ;; dmfc_rsg) echo 000130 ;;
  *) echo "unknown dataset: $1" >&2; exit 1 ;; esac; }

MF="$HOME/miniforge3"
if [ ! -d "$MF" ]; then
  wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$MF"
fi
source "$MF/etc/profile.d/conda.sh"

conda env list | grep -q '/envs/noema$' || conda create -y -n noema -c conda-forge \
  python=3.10 numpy=1.23 pandas=1.3.4 scipy scikit-learn h5py pynwb dandi
conda activate noema

pip install -q torch                        # default CUDA wheel
pip install -q --no-deps nlb_tools          # deps already provided by conda
pip install -q hydra-core wandb tqdm
pip install -q -e . --no-deps               # the noema package itself

for name in "${@:-mc_maze}"; do
  dir="data/$name"
  if [ ! -d "$dir" ]; then
    mkdir -p data
    dandi download "DANDI:$(dandiset "$name")/draft" -o data/ --existing SKIP
    mv "data/$(dandiset "$name")" "$dir"
  fi
done
mkdir -p logs
echo "conda env 'noema' ready + data in place"
