#!/usr/bin/env bash
# One-dimensional focal-plane sweeps (centre fixed, separation varied) across cameras, target ranges
# and noise levels, to find what sets the best separation, plus ablations that switch off one error
# mechanism at a time. Each scenario is a sweep folder under artifacts/focal_planes/sweeps/; reruns
# skip pairs already done.
# Usage: bash dfdd_pipeline/focal_planes/run_scenarios.sh [group ...]   (groups: aperture range noise ablation)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-./.venv/Scripts/python.exe}"
ALPHAS="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.15 1.3 1.5 1.75 2.0 2.5"
FAST="--n-cal 48 --n-test 48 --n-wide 61 --steps 300"
sweep() {  # name args...
    name=$1; shift
    "$PY" dfdd_pipeline/focal_planes/sweep_focal_planes.py --name "$name" $FAST --alphas $ALPHAS "$@" \
        > "artifacts/focal_planes/sweep_$name.log" 2>&1
    echo "done $name (exit $?)"
}
F2="--lens-mm 85 --fstop 2 --z-lo 2.0 --z-hi 4.3"
groups="${*:-aperture range noise ablation}"
for group in $groups; do
    case $group in
    aperture)  # 85 mm, 2-4.3 m, Blender-like noise: K*W from 1.1 to 6.1 px
        for f in 1.4 2 2.8 4 5.6 8; do
            sweep "ap_f${f/./p}_2to4p3" --lens-mm 85 --fstop "$f" --z-lo 2.0 --z-hi 4.3 --deltas 0 -0.2
        done ;;
    range)  # 85 mm f/2, other target ranges
        for spec in "1.0 1.6" "1.5 6.0" "3.0 6.0" "2.5 3.5" "1.2 2.0" "4.0 12.0"; do
            set -- $spec
            sweep "rg_f2_${1/./p}to${2/./p}" --lens-mm 85 --fstop 2 --z-lo "$1" --z-hi "$2" --deltas 0 -0.2
        done ;;
    noise)  # 85 mm f/2, 2-4.3 m, independent sensor noise (grey levels)
        for s in 0.25 0.5 1 2; do
            sweep "nz_f2_2to4p3_s${s/./p}" $F2 --deltas 0 -0.2 --noise "$s" --pair-correlation 0 --seeds 0 1
        done ;;
    ablation)  # 85 mm f/2, 2-4.3 m, centred pairs: which mechanism sets each side of the optimum
        # linear sensor instead of Blender's AgX view transform; with and without render noise
        sweep tn_f2_2to4p3_linear $F2 --deltas 0 --tone linear
        sweep tn_f2_2to4p3_linear_clean $F2 --deltas 0 --tone linear --noise 0 --pair-correlation 0 --quantize none
        # Gaussian defocus (the heat-equation ideal) instead of the aperture disk
        sweep ab_f2_2to4p3_gauss $F2 --deltas 0 --psf gaussian
        sweep ab_f2_2to4p3_gauss_linear_clean $F2 --deltas 0 --psf gaussian --tone linear --noise 0 --pair-correlation 0 --quantize none
        # a third, coarser pyramid scale in the estimator
        sweep ab_f2_2to4p3_3scales $F2 --deltas 0 --n-scales 3
        # invariance: f/4 on 1.0-2.15 m has the same blur span K*W (4.29 px) and ratio (2.15) as f/2 on 2.0-4.3 m
        sweep ab_f4_1to2p15_invariance --lens-mm 85 --fstop 4 --z-lo 1.0 --z-hi 2.15 --deltas 0 ;;
    esac
done
echo SCENARIOS_DONE
