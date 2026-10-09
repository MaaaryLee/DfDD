"""Check the fast slide simulator against Blender.

1. Tone curve: RMS error of the fitted per-channel map on the sharp render itself.
2. Mean image: simulator vs the seed-averaged Blender validation renders at known blur diameters;
   the residual is compared with the render noise measured from the two seeds.
3. Render noise: standard deviation, and its correlation between two renders that share a seed
   (the slide renderer uses Cycles' default seed for every image).
4. Existing slide data: simulator vs the 85 mm f/2 room-scale slide renders at every depth.

Usage: python dfdd_pipeline/focal_planes/validate_simulator.py --reference artifacts/focal_planes/reference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import LOADER_GRAY, Camera, SlideRenderer  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64) / 257.0
    return LOADER_GRAY[0] * bgr[..., 0] + LOADER_GRAY[1] * bgr[..., 1] + LOADER_GRAY[2] * bgr[..., 2]


def interior(a: np.ndarray, border: int = 24) -> np.ndarray:
    return a[..., border:-border, border:-border]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/focal_planes/reference")
    parser.add_argument("--slide-data", type=Path, default=ROOT / "artifacts/rtx3080_85mm_f2/data/zoom_range")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out_path = args.output or args.reference / "validation.json"
    report: dict = {}

    renderer = SlideRenderer(args.reference)
    meta = renderer.meta

    # 1. tone curve on the sharp hi-res render
    linear = np.load(args.reference / "sharp_linear.npy")
    png = cv2.imread(str(args.reference / "sharp.png"), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64) / 257.0
    rms = []
    for c in range(3):
        pred = np.interp(linear[..., c].ravel(), renderer.lut_x[c], renderer.lut_y[c])
        rms.append(float(np.sqrt(np.mean((pred - png[..., c].ravel()) ** 2))))
    report["tone_curve_rms_grey"] = rms
    print("tone curve RMS (grey levels, R G B):", [round(v, 3) for v in rms])

    # 2-3. validation renders
    rows = []
    blurs = sorted({rec["blur_px"] for rec in meta["validation"]})
    sim = renderer.gray_levels(torch.tensor(blurs, dtype=torch.float32)).cpu().numpy()[:, 0].astype(np.float64)
    for i, b in enumerate(blurs):
        s0 = load_gray(args.reference / f"val_b{b:g}_s0.png")
        s1 = load_gray(args.reference / f"val_b{b:g}_s1.png")
        noise = float(np.std(interior(s0 - s1)) / np.sqrt(2))
        mean = (s0 + s1) / 2
        resid = interior(sim[i] - mean)
        # Residual expected from noise alone: the two-seed mean still has noise / sqrt(2).
        rows.append({"blur_px": b, "noise_sigma_grey": noise, "texture_std_grey": float(np.std(interior(mean))),
                     "sim_minus_blender_mean_grey": float(resid.mean()), "sim_minus_blender_rms_grey": float(np.sqrt(np.mean(resid ** 2))),
                     "rms_expected_from_noise": noise / np.sqrt(2)})
        print(f"b={b:5.2f}px  noise {noise:5.2f}  texture std {rows[-1]['texture_std_grey']:6.2f}  "
              f"sim-blender mean {resid.mean():+6.2f}  rms {rows[-1]['sim_minus_blender_rms_grey']:5.2f}  (noise-only {noise / np.sqrt(2):4.2f})")
    report["validation_renders"] = rows

    # Noise correlation between renders that share a seed: compare b_a(s0)-b_b(s0) with b_a(s0)-b_b(s1).
    corr = []
    for a, b in zip(blurs[:-1], blurs[1:]):
        a0 = load_gray(args.reference / f"val_b{a:g}_s0.png")
        a1 = load_gray(args.reference / f"val_b{a:g}_s1.png")
        b0 = load_gray(args.reference / f"val_b{b:g}_s0.png")
        b1 = load_gray(args.reference / f"val_b{b:g}_s1.png")
        same = 0.5 * (np.var(interior(a0 - b0)) + np.var(interior(a1 - b1)))
        cross = 0.5 * (np.var(interior(a0 - b1)) + np.var(interior(a1 - b0)))
        na = np.var(interior(a0 - a1)) / 2
        nb = np.var(interior(b0 - b1)) / 2
        rho = (cross - same) / (2 * np.sqrt(na * nb))
        corr.append({"blur_a": a, "blur_b": b, "noise_correlation_same_seed": float(rho)})
        print(f"same-seed noise correlation between b={a:g} and b={b:g}: {rho:+.3f}")
    report["same_seed_noise_correlation"] = corr

    # 4. the existing 85 mm f/2 slide renders (focus 1.984 / 4.416 m)
    if args.slide_data.exists():
        slide_meta = json.loads((args.slide_data / "metadata.json").read_text(encoding="utf-8"))
        cam = Camera(lens_mm=slide_meta.get("lens_mm", 85.0), fstop=slide_meta["fstop"],
                     sensor_width_mm=slide_meta.get("sensor_width_mm", 36.0))
        depths = np.array(slide_meta["depths_m"])
        rows = []
        for cam_index, focus in ((0, slide_meta["focus_near_m"]), (1, slide_meta["focus_far_m"])):
            blur = cam.blur_px(focus, depths)
            sim = renderer.gray_levels(torch.tensor(blur, dtype=torch.float32)).cpu().numpy()[:, 0].astype(np.float64)
            for i, depth in enumerate(depths):
                real = load_gray(args.slide_data / f"cam_{cam_index}_500_480_{i}.png")
                resid = interior(sim[i] - real)
                rows.append({"cam": cam_index, "depth_m": float(depth), "blur_px": float(blur[i]),
                             "mean_grey": float(resid.mean()), "rms_grey": float(np.sqrt(np.mean(resid ** 2)))})
        rms = np.array([r["rms_grey"] for r in rows])
        bias = np.array([r["mean_grey"] for r in rows])
        print(f"85 mm f/2 slide renders: sim-blender RMS median {np.median(rms):.2f}, max {rms.max():.2f} grey; "
              f"mean offset median {np.median(bias):+.2f}")
        report["slide_85mm_f2"] = rows
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
