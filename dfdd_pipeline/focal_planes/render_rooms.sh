#!/usr/bin/env bash
# Render the 16 room views of the 85 mm f/2 room camera (first two views per room, as in
# indoor/camera_85mm_experiment.sh) with another focal pair, for room_check.py.
# Usage: bash dfdd_pipeline/focal_planes/render_rooms.sh NEAR FAR [OUTPUT_DIR]
set -uo pipefail
cd "$(dirname "$0")/../.."
BL="${BLENDER:-blender}"
NEAR=$1; FAR=$2
OUT="${3:-artifacts/focal_planes/rooms/focus_${NEAR}_${FAR}}"
LO=2.08; HI=4.32; FSTOP=2.0; LENS=85; SENSOR=36
blend_of() {  # room -> packed .blend (the first room lives under hello_room/)
    if [ -f "artifacts/indoor/rooms/$1/$1.blend" ]; then echo "artifacts/indoor/rooms/$1/$1.blend"; else echo "hello_room/outputs/$1/$1.blend"; fi
}
mkdir -p "$OUT"
render() {  # room exposure view...
    room=$1; exposure=$2; shift 2
    "$BL" --background "$(blend_of $room)" --python dfdd_pipeline/indoor/render_views.py -- \
        --views "dfdd_pipeline/indoor/views/$room.json" --only "$@" --output-dir "$OUT/$room" \
        --lens-mm $LENS --sensor-width-mm $SENSOR --fixed-scale 1.0 \
        --calib-focus-near "$NEAR" --calib-focus-far "$FAR" --calib-fstop $FSTOP --calib-range $LO $HI \
        --samples 4096 --exposure "$exposure" --skip-existing >> "$OUT/${room}_render.log" 2>&1
    echo "rendered $room (exit $?)"
}
render hello_room    0.0 dining_table_corner dining_cellshelf
render bathroom_s4   0.0 bath_toilet_cabinet bath_sink_wall
render bedroom_s3    0.0 bed_from_foot bed_side_table
render livingroom_s2 1.5 living_corridor_cabinet living_wall_mirror
render kitchen_s1    1.0 kitchen_sink_closeup kitchen_counter_wide
render diningroom_s5 1.0 heldout_dining_v0_chair heldout_dining_v1_chair
render bedroom_s6    0.0 heldout_bed_v0_bed heldout_bed_v1_comforter
render livingroom_s7 0.0 heldout_living_v0_cellshelf heldout_living_v1_simplebookcase
echo ROOMS_DONE
