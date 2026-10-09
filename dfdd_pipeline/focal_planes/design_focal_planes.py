"""Recommend two focus distances for a camera and a target working range (simulated, a few minutes).

The search replaces "open the camera, move a sensor, recalibrate" with the validated slide
simulator: for each candidate pair it renders a calibration sweep over the target range, calibrates
FocalSplit and scores a fresh sweep. Pairs are searched in dioptres relative to the range (see
``docs/FOCAL_PLANES.md``): centre offset delta and half-separation alpha, both in units of the
range's dioptric half-width.

Search: start from the rule of thumb (half-separation of RULE_BLUR_PX pixels of blur; centre at the
metric midpoint of the range for the mean-error objective, at the geometric mean of its ends for
the worst-depth objective), try separations around it, then centres, then refine the separation.
The rule's own pair and the "planes on the range ends" pair are scored too.

Usage:
    python dfdd_pipeline/focal_planes/design_focal_planes.py --lens-mm 85 --fstop 2 --z-lo 2.0 --z-hi 4.3
    python dfdd_pipeline/focal_planes/design_focal_planes.py --lens-mm 35 --fstop 1.4 --z-lo 0.65 --z-hi 1.35 --noise 1 --pair-correlation 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.analyze_sweep import AXIS, INK, INK2, MUTED, SERIES, SURFACE, style_axes  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import DEVICE, Camera, NoiseModel, SlideRenderer, focal_split_config  # noqa: E402
from dfdd_pipeline.focal_planes.sweep_focal_planes import evaluate_pair  # noqa: E402

RULE_BLUR_PX = 2.5  # best half-separation in blur pixels (docs/FOCAL_PLANES.md, section 4)
TOLERANCE = 0.05  # pairs within 5 % of the best objective count as equally good


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lens-mm", type=float, required=True)
    p.add_argument("--fstop", type=float, required=True)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--z-lo", type=float, required=True)
    p.add_argument("--z-hi", type=float, required=True)
    p.add_argument("--noise", type=float, default=0.05, help="Grey levels per image; 0.05 = Blender slide renders")
    p.add_argument("--pair-correlation", type=float, default=0.4)
    p.add_argument("--quantize", choices=("16bit", "8bit", "none"), default="16bit")
    p.add_argument("--objective", choices=("mean", "worst"), default="mean",
                   help="Minimise the mean or the worst-depth relative error over the range (dense)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--n-cal", type=int, default=48)
    p.add_argument("--reference", type=Path, default=ROOT / "artifacts/focal_planes/reference")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    camera = Camera(args.lens_mm, args.fstop, args.sensor_width_mm)
    k = camera.k_px_per_diopter
    u_lo, u_hi = 1 / args.z_hi, 1 / args.z_lo
    c0, w = (u_lo + u_hi) / 2, (u_hi - u_lo) / 2
    print(f"K = {k:.1f} px/D; range {args.z_lo:g}-{args.z_hi:g} m: dioptric centre {c0:.4f} D ({1 / c0:.2f} m), "
          f"half-width W = {w:.4f} D; blur span K*W = {k * w:.2f} px; ratio {args.z_hi / args.z_lo:.2f}", flush=True)

    renderer = SlideRenderer(args.reference)
    config = focal_split_config()
    noise = NoiseModel(args.noise, args.pair_correlation, args.quantize)
    cal = np.linspace(args.z_lo, args.z_hi, args.n_cal)
    test = np.linspace(args.z_lo, args.z_hi, 48)
    wide = 1 / np.linspace(c0 + 1.8 * w, max(c0 - 1.8 * w, 0.02), 61)
    field = "mean_rel" if args.objective == "mean" else "max_rel"
    cache: dict = {}

    def score(alpha: float, delta: float) -> dict | None:
        alpha, delta = round(alpha, 3), round(delta, 3)
        if (alpha, delta) in cache:
            return cache[(alpha, delta)]
        near_u, far_u = c0 + (delta + alpha) * w, c0 + (delta - alpha) * w
        if far_u <= 0.005 or alpha <= 0:
            cache[(alpha, delta)] = None
            return None
        gen = torch.Generator(device=DEVICE).manual_seed(11)
        t0 = time.time()
        rec = evaluate_pair(renderer, camera, config, 1 / near_u, 1 / far_u, cal, {"test": test, "wide": wide}, noise,
                            args.steps, gen, args.z_lo, args.z_hi)
        rec.update(alpha=alpha, delta=delta)
        cache[(alpha, delta)] = rec
        t = rec["test"]["0.0"]
        print(f"  alpha {alpha:.3f} delta {delta:+.2f}: focus {rec['near_m']:.3f} / {rec['far_m']:.3f} m -> "
              f"mean {100 * t['mean_rel']:.2f}%, worst {100 * t['max_rel']:.1f}% ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
        return rec

    def value(rec):
        return np.inf if rec is None else rec["test"]["0.0"][field]

    def best_alpha(alphas, delta):
        recs = [(a, score(a, delta)) for a in alphas]
        return min(recs, key=lambda item: value(item[1]))

    alpha0 = RULE_BLUR_PX / (k * w)
    # Pair centre (dioptres): metric midpoint of the range for the mean error, geometric mean of the
    # range ends for the worst depth (docs/FOCAL_PLANES.md, section 4).
    centre = 2.0 / (args.z_lo + args.z_hi) if args.objective == "mean" else 1.0 / np.sqrt(args.z_lo * args.z_hi)
    delta0 = (centre - c0) / w
    print(f"rule of thumb: alpha {alpha0:.3f}, centre {delta0:+.3f} W", flush=True)
    rule = score(alpha0, delta0)
    print("separation search around the rule", flush=True)
    a_best, _ = best_alpha([alpha0 * f for f in (0.6, 0.8, 1.0, 1.25, 1.6)], delta0)
    a_best, _ = best_alpha([a_best * f for f in (0.9, 1.0, 1.1)], delta0)
    print("centre search", flush=True)
    deltas = [delta0 + d for d in (-0.15, -0.075, 0.0, 0.075, 0.15)]
    d_best = min(deltas, key=lambda d: value(score(a_best, d)))
    print("final separation refinement", flush=True)
    a_best, best = best_alpha([a_best * f for f in (0.93, 1.0, 1.07)], d_best)
    print("planes on the range ends", flush=True)
    ends = score(1.0, 0.0)

    # The error surface is flat near its minimum: among the pairs within TOLERANCE of the best
    # objective, recommend the one that is best on the other measure (worst depth for the mean
    # objective, mean for the worst-depth objective).
    other = "max_rel" if field == "mean_rel" else "mean_rel"
    evaluated = [r for r in cache.values() if r is not None]
    near_best = [r for r in evaluated if r["test"]["0.0"][field] <= (1 + TOLERANCE) * best["test"]["0.0"][field]]
    best = min(near_best, key=lambda r: r["test"]["0.0"][other])

    bt, et = best["test"]["0.0"], ends["test"]["0.0"]
    summary = {
        "camera": camera.__dict__, "k_px_per_diopter": k, "z_lo": args.z_lo, "z_hi": args.z_hi, "blur_span_px": k * w,
        "noise": noise.__dict__, "objective": args.objective,
        "recommended": {"near_m": best["near_m"], "far_m": best["far_m"], "alpha": best["alpha"], "delta": best["delta"],
                        "mean_rel": bt["mean_rel"], "max_rel": bt["max_rel"], "mean_rel_90": best["test"]["0.9"]["mean_rel"]},
        "range_ends": {"near_m": ends["near_m"], "far_m": ends["far_m"], "mean_rel": et["mean_rel"], "max_rel": et["max_rel"],
                       "mean_rel_90": ends["test"]["0.9"]["mean_rel"]},
        "rule_of_thumb": None if rule is None else {
            "near_m": rule["near_m"], "far_m": rule["far_m"], "alpha": rule["alpha"], "delta": rule["delta"],
            "mean_rel": rule["test"]["0.0"]["mean_rel"], "max_rel": rule["test"]["0.0"]["max_rel"]},
        "evaluated": [{k2: v for k2, v in r.items() if k2 not in ("test", "wide")} | {"mean_rel": r["test"]["0.0"]["mean_rel"],
                      "max_rel": r["test"]["0.0"]["max_rel"]} for r in cache.values() if r is not None],
    }
    out = args.output or ROOT / "artifacts/focal_planes/designs" / f"{args.lens_mm:g}mm_f{args.fstop:g}_{args.z_lo:g}-{args.z_hi:g}m.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axvspan(args.z_lo, args.z_hi, color=SERIES[0], alpha=0.07, lw=0)
    ax.axhline(10, color=AXIS, lw=1)
    for i, (label, rec) in enumerate((("recommended", best), ("planes on the range ends", ends))):
        depths = np.array(rec["wide"]["depths_m"])
        rel = 100 * np.array(rec["wide"]["0.0"]["per_depth_mae_m"]) / depths
        ax.plot(depths, rel, color=SERIES[i], lw=2, label=f"{label}: focus {rec['near_m']:.2f} / {rec['far_m']:.2f} m, "
                f"mean {100 * rec['test']['0.0']['mean_rel']:.2f}% in range")
        for plane in (rec["near_m"], rec["far_m"]):
            ax.plot(plane, 0, marker="|", ms=14, mew=2, color=SERIES[i], clip_on=False)
    ax.set_ylim(0, 25)
    ax.set_xlabel("true depth (m); shaded = target range, ticks = focal planes")
    ax.set_ylabel("relative depth error, % (dense)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center")
    ax.set_title(f"{args.lens_mm:g} mm f/{args.fstop:g}, target {args.z_lo:g}-{args.z_hi:g} m (simulated slide)",
                 loc="left", color=INK, fontsize=11)
    fig.savefig(out.with_suffix(".png"), dpi=150, facecolor=SURFACE)

    if rule is not None:
        rt = rule["test"]["0.0"]
        print(f"\nRule of thumb: near {rule['near_m']:.3f} m, far {rule['far_m']:.3f} m: "
              f"mean {100 * rt['mean_rel']:.2f}%, worst {100 * rt['max_rel']:.1f}%")
    print(f"Recommended focus: near {best['near_m']:.3f} m, far {best['far_m']:.3f} m "
          f"(alpha {best['alpha']:.2f}, centre {best['delta']:+.2f} W): mean {100 * bt['mean_rel']:.2f}%, worst {100 * bt['max_rel']:.1f}% "
          f"| planes on the range ends ({ends['near_m']:.2f} / {ends['far_m']:.2f} m): mean {100 * et['mean_rel']:.2f}%, worst {100 * et['max_rel']:.1f}%")
    print("wrote", out, "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
