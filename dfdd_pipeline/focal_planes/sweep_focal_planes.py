"""Grid search over focal-plane pairs for a target working range, with the fast slide simulator.

Every pair is the virtual equivalent of re-focusing the two sensors and re-calibrating: render a
calibration slide sweep over the target range [z_lo, z_hi] (uniform in metres, as the pipeline
does), calibrate FocalSplit on it, then score a fresh sweep (new noise) inside the range and a wider
sweep that shows where the working range actually ends.

Pairs are parameterised in dioptres relative to the target range: with c0 and W the dioptric centre
and half-width of [z_lo, z_hi], the focal planes sit at c0 + delta*W +- alpha*W. ``alpha = 1,
delta = 0`` puts them exactly on the range ends; ``alpha < 1`` moves both inward. ``--pairs`` adds
explicit (near, far) pairs in metres.

Usage:
    python dfdd_pipeline/focal_planes/sweep_focal_planes.py --name f2_2to4p3 --lens-mm 85 --fstop 2 \
        --z-lo 2.0 --z-hi 4.3 --alphas 0.3 0.5 0.7 0.9 1.1 --deltas -0.2 0 0.2
Results: artifacts/focal_planes/sweeps/<name>/results.jsonl (one JSON record per pair; reruns skip
pairs already present).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import (  # noqa: E402
    DEVICE, Camera, NoiseModel, SlideRenderer, calibrate, focal_split_config, per_depth_errors, predict,
    slide_sweep, summarize_range)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--reference", type=Path, default=ROOT / "artifacts/focal_planes/reference")
    p.add_argument("--tone", choices=("agx", "linear"), default="agx")
    p.add_argument("--psf", choices=("disk", "gaussian"), default="disk", help="Defocus kernel (gaussian = ablation)")
    p.add_argument("--lens-mm", type=float, default=85.0)
    p.add_argument("--fstop", type=float, default=2.0)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--z-lo", type=float, required=True)
    p.add_argument("--z-hi", type=float, required=True)
    p.add_argument("--alphas", type=float, nargs="*", default=[])
    p.add_argument("--deltas", type=float, nargs="*", default=[0.0])
    p.add_argument("--pairs", type=float, nargs="*", default=[], help="Explicit near far near far ... (metres)")
    p.add_argument("--noise", type=float, default=0.05, help="White noise per image (grey levels, 0-255)")
    p.add_argument("--pair-correlation", type=float, default=0.4, help="Noise correlation of the two focus images")
    p.add_argument("--quantize", choices=("16bit", "8bit", "none"), default="16bit")
    p.add_argument("--n-cal", type=int, default=71)
    p.add_argument("--n-test", type=int, default=71)
    p.add_argument("--n-wide", type=int, default=81)
    p.add_argument("--wide", type=float, default=1.8, help="Wide sweep covers c0 +- wide*W (dioptres)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--schedule", choices=("cosine", "constant"), default="cosine",
                   help="Learning-rate schedule; the trainer's constant rate never settles (see simulator.calibrate)")
    p.add_argument("--n-scales", type=int, default=2, help="FocalSplit pyramid scales (the pipeline uses 2)")
    p.add_argument("--seeds", type=int, nargs="*", default=[0], help="Noise realisations per pair")
    return p.parse_args()


def designs(args, c0: float, w: float) -> list[dict]:
    out = []
    for delta in args.deltas:
        for alpha in args.alphas:
            near_u, far_u = c0 + (delta + alpha) * w, c0 + (delta - alpha) * w
            out.append({"alpha": alpha, "delta": delta, "near_u": near_u, "far_u": far_u})
    for near, far in zip(args.pairs[::2], args.pairs[1::2]):
        near_u, far_u = 1.0 / near, (1.0 / far if far < math.inf else 0.0)
        out.append({"alpha": (near_u - far_u) / (2 * w), "delta": ((near_u + far_u) / 2 - c0) / w,
                    "near_u": near_u, "far_u": far_u, "explicit": True})
    return out


def key(d: dict, seed: int) -> str:
    return f"{d['near_u']:.6f}_{d['far_u']:.6f}_{seed}"


def evaluate_pair(renderer, camera, config, near_m: float, far_m: float, cal_depths, test_sets: dict, noise,
                  steps: int, gen, z_lo: float, z_hi: float, schedule: str = "cosine") -> dict:
    """Calibrate on a simulated slide sweep of ``cal_depths`` and score fresh sweeps of each test set."""
    near_u, far_u = 1.0 / near_m, 1.0 / far_m
    ip, im, z = slide_sweep(renderer, camera, near_m, far_m, cal_depths, noise, gen)
    result = calibrate(ip, im, z, config, steps=steps, init="physics", b_value=0.5 * (near_u + far_u), schedule=schedule)
    del ip, im
    record = {"near_m": near_m, "far_m": far_m, "b_half_px": camera.k_px_per_diopter * 0.5 * (near_u - far_u),
              "final_loss": result.losses[-1],
              "A": [float(a.x.detach().mean()) for a in result.model.A_list],
              "B": [float(b.x.detach().mean()) for b in result.model.B_list]}
    for label, depths in test_sets.items():
        tp, tm, _ = slide_sweep(renderer, camera, near_m, far_m, depths, noise, gen)
        z_pred, conf = predict(result.model, tp, tm)
        del tp, tm
        block = {}
        order = np.argsort(depths)
        for s, values in per_depth_errors(z_pred, conf, depths).items():
            summary = summarize_range(depths, values["mae_m"], z_lo, z_hi)
            summary["slope"] = float(np.polyfit(depths, values["median_pred_m"], 1)[0])
            summary["per_depth_mae_m"] = values["mae_m"][order].round(5).tolist()
            summary["per_depth_median_m"] = values["median_pred_m"][order].round(5).tolist()
            block[str(s)] = summary
        block["depths_m"] = np.sort(depths).round(5).tolist()
        record[label] = block
    return record


def main() -> None:
    args = parse_args()
    out_dir = ROOT / "artifacts/focal_planes/sweeps" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = Camera(args.lens_mm, args.fstop, args.sensor_width_mm)
    noise = NoiseModel(args.noise, args.pair_correlation, args.quantize)
    u_lo, u_hi = 1.0 / args.z_hi, 1.0 / args.z_lo
    c0, w = (u_lo + u_hi) / 2, (u_hi - u_lo) / 2
    meta = {"camera": camera.__dict__, "k_px_per_diopter": camera.k_px_per_diopter, "z_lo": args.z_lo, "z_hi": args.z_hi,
            "c0_diopter": c0, "w_diopter": w, "range_blur_span_px": camera.k_px_per_diopter * w,
            "noise": noise.__dict__, "tone": args.tone, "psf": args.psf, "n_cal": args.n_cal, "n_test": args.n_test,
            "n_wide": args.n_wide, "wide": args.wide, "steps": args.steps, "n_scales": args.n_scales,
            "schedule": args.schedule}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"K = {camera.k_px_per_diopter:.2f} px/D, range {args.z_lo}-{args.z_hi} m: c0 = {c0:.4f} D, W = {w:.4f} D, "
          f"K*W = {camera.k_px_per_diopter * w:.2f} px", flush=True)

    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done.add(key(rec, rec["seed"]))

    renderer = SlideRenderer(args.reference, tone=args.tone, psf=args.psf)
    config = focal_split_config(n_scales=args.n_scales)
    cal_depths = np.linspace(args.z_lo, args.z_hi, args.n_cal)
    test_depths = np.linspace(args.z_lo, args.z_hi, args.n_test)
    wide_u = np.linspace(c0 + args.wide * w, max(c0 - args.wide * w, 0.02), args.n_wide)
    wide_depths = 1.0 / wide_u

    for d in designs(args, c0, w):
        if d["far_u"] <= 0:
            print(f"skip alpha={d['alpha']:.2f} delta={d['delta']:+.2f}: far plane beyond infinity", flush=True)
            continue
        near_m, far_m = 1.0 / d["near_u"], 1.0 / d["far_u"]
        for seed in args.seeds:
            if key(d, seed) in done:
                continue
            t0 = time.time()
            gen = torch.Generator(device=DEVICE).manual_seed(1000 * seed + 1)
            record = {**d, "seed": seed, **evaluate_pair(renderer, camera, config, near_m, far_m, cal_depths,
                                                         {"test": test_depths, "wide": wide_depths}, noise,
                                                         args.steps, gen, args.z_lo, args.z_hi, args.schedule)}
            record["seconds"] = round(time.time() - t0, 1)
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            t, s9 = record["test"]["0.0"], record["test"]["0.9"]
            print(f"alpha={d['alpha']:.2f} delta={d['delta']:+.2f} focus {near_m:.3f}/{far_m:.3f} m (b_h {record['b_half_px']:.2f} px) "
                  f"seed {seed}: dense rel {t['mean_rel'] * 100:5.2f}% max {t['max_rel'] * 100:5.1f}% cov {t['coverage']:.2f} | "
                  f"90% rel {s9['mean_rel'] * 100:5.2f}% max {s9['max_rel'] * 100:5.1f}% | {record['seconds']}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
