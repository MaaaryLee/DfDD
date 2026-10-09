"""Confirm simulator predictions with real Blender slide sweeps.

For each focal-plane pair: render the calibration slide with ``calibration/render_checkerboard_slide.py``
(same settings as the room-scale 85 mm run: 2048 samples, 16-bit, no denoising or adaptive sampling,
fractal texture, constant exposure), calibrate FocalSplit on the Blender images, and compare the
per-depth error with a simulated sweep of the same pair. An existing rendered sweep can be listed
with ``--existing near far path``.

Usage:
    python dfdd_pipeline/focal_planes/blender_check.py --name f2_2p08to4p32 --pairs 2.24 3.50 2.3 4.0 \
        --existing 1.984 4.416 artifacts/rtx3080_85mm_f2/data/zoom_range
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.common.platform_support import find_blender  # noqa: E402
from dfdd_pipeline.focal_planes.check_calibration import load_blender_sweep  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import (  # noqa: E402
    DEVICE, Camera, NoiseModel, SlideRenderer, calibrate, focal_split_config, per_depth_errors, predict,
    slide_sweep, summarize_range)


def render(args, near: float, far: float, out: Path, depths: np.ndarray) -> None:
    if (out / "metadata.json").exists():
        return
    command = [find_blender(args.blender), "--background", "--python",
               str(ROOT / "dfdd_pipeline/calibration/render_checkerboard_slide.py"), "--",
               "--output-dir", str(out), "--depths", *[f"{d:.6f}" for d in depths],
               "--focus-near", str(near), "--focus-far", str(far), "--fstop", str(args.fstop),
               "--lens-mm", str(args.lens_mm), "--sensor-width-mm", str(args.sensor_width_mm),
               "--samples", str(args.samples), "--color-depth", "16", "--no-adaptive-sampling", "--no-denoising",
               "--texture", "fractal", "--constant-exposure", "--device", args.device]
    log = out.parent / "render.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(command, check=True, stdout=fh, stderr=subprocess.STDOUT)


def evaluate(model, i_plus, i_minus, depths, z_lo, z_hi) -> dict:
    z_pred, conf = predict(model, i_plus, i_minus)
    out = {}
    for s, values in per_depth_errors(z_pred, conf, depths).items():
        summary = summarize_range(depths, values["mae_m"], z_lo, z_hi)
        summary["per_depth_mae_m"] = values["mae_m"].round(5).tolist()
        out[str(s)] = summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--pairs", type=float, nargs="*", default=[])
    parser.add_argument("--existing", nargs=3, action="append", default=[], metavar=("NEAR", "FAR", "DIR"))
    parser.add_argument("--z-lo", type=float, default=2.08)
    parser.add_argument("--z-hi", type=float, default=4.32)
    parser.add_argument("--n", type=int, default=71)
    parser.add_argument("--lens-mm", type=float, default=85.0)
    parser.add_argument("--fstop", type=float, default=2.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--pair-correlation", type=float, default=0.4)
    parser.add_argument("--blender")
    parser.add_argument("--device", default="optix")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/focal_planes/reference")
    args = parser.parse_args()

    base = ROOT / "artifacts/focal_planes/blender" / args.name
    base.mkdir(parents=True, exist_ok=True)
    depths = np.array([round(float(v), 6) for v in np.linspace(args.z_lo, args.z_hi, args.n)])
    jobs = [(float(n), float(f), base / f"focus_{float(n):.3f}_{float(f):.3f}" / "zoom_range") for n, f in zip(args.pairs[::2], args.pairs[1::2])]
    jobs += [(float(n), float(f), Path(d)) for n, f, d in args.existing]
    for near, far, data in jobs:
        print(f"render {near}/{far} -> {data}", flush=True)
        render(args, near, far, data, depths)

    camera = Camera(args.lens_mm, args.fstop, args.sensor_width_mm)
    renderer = SlideRenderer(args.reference)
    config = focal_split_config()
    noise = NoiseModel(args.noise, args.pair_correlation, "16bit")
    results = []
    for near, far, data in jobs:
        ip, im, z, meta = load_blender_sweep(data)
        d = z.cpu().numpy().ravel().astype(np.float64)
        u_mid = 0.5 * (1 / near + 1 / far)
        blender = calibrate(ip, im, z, config, steps=args.steps, init="physics", b_value=u_mid, schedule=args.schedule)
        rec = {"near_m": near, "far_m": far, "data": str(data), "depths_m": d.round(5).tolist(),
               "blender": evaluate(blender.model, ip, im, d, args.z_lo, args.z_hi)}
        gen = torch.Generator(device=DEVICE).manual_seed(7)
        sp, sm, sz = slide_sweep(renderer, camera, near, far, d, noise, gen)
        sim = calibrate(sp, sm, sz, config, steps=args.steps, init="physics", b_value=u_mid, schedule=args.schedule)
        rec["sim"] = evaluate(sim.model, sp, sm, d, args.z_lo, args.z_hi)
        # Cross-check: the Blender-calibrated model applied to simulated images and vice versa.
        rec["blender_model_on_sim"] = evaluate(blender.model, sp, sm, d, args.z_lo, args.z_hi)
        results.append(rec)
        b, s = rec["blender"], rec["sim"]
        print(f"{near:.3f}/{far:.3f} m: Blender dense {100 * b['0.0']['mean_rel']:.2f}% (worst {100 * b['0.0']['max_rel']:.1f}%), "
              f"90% {100 * b['0.9']['mean_rel']:.2f}% | sim dense {100 * s['0.0']['mean_rel']:.2f}% (worst {100 * s['0.0']['max_rel']:.1f}%), "
              f"90% {100 * s['0.9']['mean_rel']:.2f}%", flush=True)
        del ip, im, sp, sm
        torch.cuda.empty_cache()
    (base / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", base / "results.json")


if __name__ == "__main__":
    main()
