#!/usr/bin/env bash
# macOS / Linux counterpart of run_rtx3080.ps1.
set -euo pipefail

cd "$(dirname "$0")"

git submodule update --init --recursive

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -x ".venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv .venv
fi
VENV_PYTHON=".venv/bin/python"

"$VENV_PYTHON" rtx3080_pipeline/prepare_vendor.py
"$VENV_PYTHON" -m pip install --upgrade pip

# Apple silicon uses the default wheels (MPS); elsewhere prefer the CUDA build.
if [ "$(uname -s)" = "Darwin" ]; then
    "$VENV_PYTHON" -m pip install torch torchvision
else
    "$VENV_PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
"$VENV_PYTHON" -m pip install -r requirements-rtx3080.txt

"$VENV_PYTHON" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
"$VENV_PYTHON" rtx3080_pipeline/run_pipeline.py "$@"
