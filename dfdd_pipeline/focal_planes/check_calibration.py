"""Check the fast calibration against the pipeline, and simulated slides against Blender's.

For one existing Blender slide sweep (default: 85 mm f/2, focus 1.984 / 4.416 m, 2.08-4.32 m):

* ``blender/default``: the Blender images, SpiderCam's initialisation, the pipeline's step count;
  should reproduce the pipeline calibration (``RUN_SUMMARY.md``).
* ``blender/physics``: the same images with ``physics_init`` and fewer steps.
* ``sim/physics``: simulated images of the same sweep (render noise and same-seed correlation as
  measured by ``validate_simulator.py``) with ``physics_init``.

Every variant is scored on the frames it was trained on, as the pipeline's summary is.

Usage: python dfdd_pipeline/focal_planes/check_calibration.py --data artifacts/rtx3080_85mm_f2/data/zoom_range
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import (  # noqa: E402
    DEVICE, LOADER_GRAY, Camera, NoiseModel, SlideRenderer, calibrate, focal_split_config, per_depth_errors,
    predict, slide_sweep, summarize_range)


def load_blender_sweep(data_dir: Path, full_precision: bool = True):
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    depths = np.array(meta["depths_m"], dtype=np.float64)
    planes = []
    for cam in (1, 0):  # I_plus = cam_1 (far focus), I_minus = cam_0 (near focus)
        images = []
        for i in range(len(depths)):
            raw = cv2.imread(str(data_dir / f"cam_{cam}_500_480_{i}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
            bgr = raw / 257.0 if full_precision else np.round(raw / 256.0)
            images.append(LOADER_GRAY[0] * bgr[..., 0] + LOADER_GRAY[1] * bgr[..., 1] + LOADER_GRAY[2] * bgr[..., 2])
        planes.append(torch.from_numpy(np.stack(images)[:, None] / 256.0).to(DEVICE))
    z = torch.as_tensor(depths, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    return planes[0], planes[1], z, meta


def score(model, i_plus, i_minus, depths, z_lo, z_hi) -> dict:
    z_pred, conf = predict(model, i_plus, i_minus)
    errors = per_depth_errors(z_pred, conf, depths)
    out = {}
    for s, values in errors.items():
        summary = summarize_range(depths, values["mae_m"], z_lo, z_hi)
        slope = float(np.polyfit(depths, values["median_pred_m"], 1)[0])
        out[str(s)] = {**summary, "slope": slope, "per_depth_mae_m": values["mae_m"].round(5).tolist()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/rtx3080_85mm_f2/data/zoom_range")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/focal_planes/reference")
    parser.add_argument("--default-steps", type=int, default=2000)
    parser.add_argument("--physics-steps", type=int, nargs="+", default=[200, 400, 800])
    parser.add_argument("--noise", type=float, default=0.05, help="Render noise (grey levels) of the simulated sweep")
    parser.add_argument("--pair-correlation", type=float, default=0.4)
    parser.add_argument("--skip-default", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    i_plus, i_minus, z, meta = load_blender_sweep(args.data)
    depths = z.cpu().numpy().ravel().astype(np.float64)
    z_lo, z_hi = float(depths.min()), float(depths.max())
    u_mid = 0.5 * (1.0 / meta["focus_near_m"] + 1.0 / meta["focus_far_m"])
    config = focal_split_config()
    results = {"data": str(args.data), "focus_m": [meta["focus_near_m"], meta["focus_far_m"]], "fstop": meta["fstop"]}

    def run(label, ip, im, init, steps):
        t0 = time.time()
        result = calibrate(ip, im, z, config, steps=steps, init=init, b_value=u_mid)
        scored = score(result.model, ip, im, depths, z_lo, z_hi)
        results[label] = {"steps": steps, "seconds": round(time.time() - t0, 1), "final_loss": result.losses[-1], **scored}
        d, s = scored["0.0"], scored["0.9"]
        print(f"{label:24s} {steps:5d} steps {time.time() - t0:6.1f}s  loss {result.losses[-1]:.4f}  "
              f"dense MAE {d['mae_m'] * 100:5.1f} cm rel {d['mean_rel'] * 100:4.1f}% WR {d['working_range_m']} slope {d['slope']:.3f} | "
              f"90% MAE {s['mae_m'] * 100:5.1f} cm rel {s['mean_rel'] * 100:4.1f}%", flush=True)

    if not args.skip_default:
        run("blender/default", i_plus, i_minus, "default", args.default_steps)
    for steps in args.physics_steps:
        run(f"blender/physics_{steps}", i_plus, i_minus, "physics", steps)

    renderer = SlideRenderer(args.reference)
    camera = Camera(lens_mm=meta.get("lens_mm", 35.0), fstop=meta["fstop"], sensor_width_mm=meta.get("sensor_width_mm", 36.0))
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    noise = NoiseModel(sigma=args.noise, pair_correlation=args.pair_correlation, quantize="16bit")
    s_plus, s_minus, _ = slide_sweep(renderer, camera, meta["focus_near_m"], meta["focus_far_m"], depths, noise, generator)
    for steps in args.physics_steps:
        run(f"sim/physics_{steps}", s_plus, s_minus, "physics", steps)

    out = args.output or ROOT / "artifacts/focal_planes" / f"check_{args.data.parent.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
