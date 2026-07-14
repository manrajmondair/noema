#!/usr/bin/env bash
# One-time setup on a FarmShare LOGIN node (rice.stanford.edu): build the
# environment and fetch data while internet is available — compute nodes may not
# have it. Run this before submitting any GPU job.
#   Usage: scripts/farmshare_setup.sh [dataset] [dandiset]   (default: mc_maze 000128)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-mc_maze}"
DANDISET="${2:-000128}"

module load python 2>/dev/null || true          # best-effort; ignore if no module system
PY=$(command -v python3.11 || command -v python3)
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 9), f"need Python >=3.9, found {sys.version.split()[0]}; try `module avail python`"
EOF
echo "using $("$PY" --version)"

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[data,train]"               # torch pulls the default CUDA wheel here

DATA_DIR="data/${DATASET}"
if [ ! -d "$DATA_DIR" ]; then
  mkdir -p data
  dandi download "DANDI:${DANDISET}/draft" -o data/
  mv "data/${DANDISET}" "$DATA_DIR"              # dandi nests under the dandiset id
fi
mkdir -p logs
echo "setup complete — submit training with: sbatch scripts/farmshare_nlb.sbatch"
