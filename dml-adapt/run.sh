#!/usr/bin/env bash
# Bridge runner: edit a script in WSL, run it with the Windows python that has
# torch-directml installed (torch-directml is Windows-only, so it cannot live in
# WSL's own python).
#
# === ADJUST THESE TWO PATHS FOR YOUR MACHINE ===
#   CONDA_DML  -> your Windows conda env that has torch + torch-directml
#   WIN_LOCAL  -> a Windows-local dir to copy the script into (UNC paths from WSL
#                 are not reachable by Windows python, so we stage a copy)
CONDA_DML="d:/Users/rickq/miniconda3/Scripts/conda.exe"
WIN_LOCAL="/mnt/c/Users/rickq/dml-adapt"

set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WIN_LOCAL"
# copy the .py files that changed (-u: only if newer)
cp -u "$SRC_DIR"/*.py "$WIN_LOCAL"/ 2>/dev/null || true

SCRIPT="${1:-train_gpt_ch04.py}"
echo "[run.sh] executing $SCRIPT in conda env 'dml' ..."
powershell.exe -NoProfile -Command \
  "& '$CONDA_DML' run -n dml python 'C:/Users/rickq/dml-adapt/$SCRIPT'"
