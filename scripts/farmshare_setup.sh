#!/usr/bin/env bash
# One-time setup on a FarmShare LOGIN node (rice.stanford.edu): build the
# environment and fetch data while internet is available — compute nodes may not
# have it. Pass dataset names (default mc_maze); list several to prep pretraining.
#   scripts/farmshare_setup.sh                              # just mc_maze
#   scripts/farmshare_setup.sh mc_maze mc_rtt area2_bump    # for cross-subject pretraining
set -euo pipefail
cd "$(dirname "$0")/.."

dandiset() { case "$1" in
  mc_maze) echo 000128 ;; mc_rtt) echo 000129 ;;
  area2_bump) echo 000127 ;; dmfc_rsg) echo 000130 ;;
  *) echo "unknown dataset: $1" >&2; exit 1 ;; esac; }

module load python 2>/dev/null || true          # best-effort; ignore if no module system
PY=$(command -v python3.11 || command -v python3)
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 9), f"need Python >=3.9, found {sys.version.split()[0]}; try `module avail python`"
EOF
echo "using $("$PY" --version)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q -e ".[train]"                    # torch (CUDA wheel) + hydra + wandb + tqdm
pip install -q numpy scipy pandas h5py pynwb dandi   # modern sci stack, all prebuilt for 3.12
pip install -q --no-deps nlb_tools              # its stale pandas pin would force a source build

for name in "${@:-mc_maze}"; do
  dir="data/$name"
  if [ ! -d "$dir" ]; then
    mkdir -p data
    dandi download "DANDI:$(dandiset "$name")/draft" -o data/
    mv "data/$(dandiset "$name")" "$dir"        # dandi nests under the dandiset id
  fi
done
mkdir -p logs
echo "setup complete — submit training with: sbatch scripts/farmshare_nlb.sbatch"
