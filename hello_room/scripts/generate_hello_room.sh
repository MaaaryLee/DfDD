#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace/work/infinigen"
out="$workspace/outputs/hello_room"
mkdir -p "$out/logs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

if [[ ! -f "$out/coarse/scene.blend" ]]; then
    .venv/bin/python -m infinigen_examples.generate_indoors \
        --seed 0 --task coarse --output_folder "$out/coarse" \
        -g fast_solve.gin singleroom.gin \
        -p compose_indoors.terrain_enabled=False \
        'restrict_solving.restrict_parent_rooms=["DiningRoom"]' \
        2>&1 | tee "$out/logs/coarse.log"
fi

.venv/bin/python "$workspace/scripts/render_hello_room.py" \
    --seed 0 --task render --input_folder "$out/coarse" \
    --output_folder "$out/frames" \
    -p configure_render_cycles.num_samples=256 \
    configure_render_cycles.denoise=True \
    configure_render_cycles.adaptive_threshold=0.01 \
    2>&1 | tee "$out/logs/render.log"
