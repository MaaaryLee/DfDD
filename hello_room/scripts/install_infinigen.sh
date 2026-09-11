#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null
command -v git >/dev/null
repo="$workspace/work/infinigen"
if [[ ! -d "$repo" ]]; then
    mkdir -p "$workspace/work"
    git clone --depth 1 --branch v1.19.0 https://github.com/princeton-vl/infinigen.git "$repo"
fi
cd "$repo"
expected_commit=01c39c7f7adcf7363ccbcc57c64410c69f4a4e7c
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
    echo "Expected Infinigen v1.19.0 at $expected_commit; refusing to change an existing checkout." >&2
    exit 1
fi
git submodule update --init --depth 1 infinigen/infinigen_gpl infinigen/OcMesher
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi
export INFINIGEN_MINIMAL_INSTALL=True
export NPY_NUM_BUILD_JOBS=8
mkdir -p "$workspace/outputs/hello_room/logs"
uv pip install --python .venv/bin/python \
    --build-constraint ../../scripts/infinigen-build-constraints.txt \
    --constraint ../../requirements-lock.txt \
    -e . pyyaml > "$workspace/outputs/hello_room/logs/install.log" 2>&1
.venv/bin/python -c 'import bpy, gin, skimage, scipy, shapely; print("Blender:", bpy.app.version_string, "scikit-image:", skimage.__version__)'
