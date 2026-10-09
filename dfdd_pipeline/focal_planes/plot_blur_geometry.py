"""Explanatory figure: blur of the two images against depth for two focal pairs.

The depth axis is linear in dioptres (labelled in metres), so each image's blur diameter is a
straight V. With the planes on the range ends, an object at either end is a disk of 2*K*W pixels in
the other image; the rule's pair keeps that at 2*b = 5 px.

Usage: python dfdd_pipeline/focal_planes/plot_blur_geometry.py [--lens-mm 85 --fstop 2 --z-lo 2 --z-hi 4.3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.analyze_sweep import AXIS, INK, INK2, MUTED, SERIES, SURFACE, style_axes  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import Camera  # noqa: E402

RULE_BLUR_PX = 2.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-mm", type=float, default=85.0)
    parser.add_argument("--fstop", type=float, default=2.0)
    parser.add_argument("--z-lo", type=float, default=2.0)
    parser.add_argument("--z-hi", type=float, default=4.3)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/focal_planes/blur_geometry.png")
    args = parser.parse_args()
    cam = Camera(args.lens_mm, args.fstop)
    k = cam.k_px_per_diopter
    u_lo, u_hi = 1 / args.z_hi, 1 / args.z_lo
    c0, w = (u_lo + u_hi) / 2, (u_hi - u_lo) / 2
    rule_c, rule_h = np.sqrt(u_lo * u_hi), RULE_BLUR_PX / k
    pairs = [("Planes on the range ends", c0, w), ("Rule: half-separation 2.5 px, centre 1/sqrt(z_lo z_hi)", rule_c, rule_h)]

    u = np.linspace(u_lo - 0.6 * w, u_hi + 0.6 * w, 400)
    ticks_m = [z for z in (1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 8) if u.min() <= 1 / z <= u.max()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, c, h) in zip(axes, pairs):
        style_axes(ax)
        near_u, far_u = c + h, c - h
        ax.axvspan(u_lo, u_hi, color=SERIES[0], alpha=0.07, lw=0)
        ax.plot(u, k * np.abs(u - near_u), color=SERIES[0], lw=2, label=f"near-focus image (focus {1 / near_u:.2f} m)")
        ax.plot(u, k * np.abs(u - far_u), color=SERIES[1], lw=2, label=f"far-focus image (focus {1 / far_u:.2f} m)")
        ax.axhline(2 * RULE_BLUR_PX, color=AXIS, lw=1)
        ax.text(u.min(), 2 * RULE_BLUR_PX, "5 px disk ", color=MUTED, fontsize=8, va="bottom", ha="right")
        for plane, other in ((near_u, far_u), (far_u, near_u)):
            ax.plot(plane, k * abs(plane - other), marker="o", ms=7, color=INK, mec=SURFACE, mew=1.5)
        ax.annotate(f"{2 * k * h:.1f} px", (near_u, 2 * k * h), textcoords="offset points", xytext=(6, 4), color=INK2, fontsize=9)
        ends = max(k * abs(u_hi - far_u), k * abs(u_lo - near_u))
        ax.set_title(f"{title}\nlargest blur inside the range {ends:.1f} px", loc="left", color=INK, fontsize=10)
        ax.set_xticks([1 / z for z in ticks_m])
        ax.set_xticklabels([f"{z:g}" for z in ticks_m])
        ax.set_xlim(u.max(), u.min())  # near on the left, as distance grows to the right
        ax.set_xlabel("depth (m), spaced linearly in dioptres; shaded = target range")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center")
    axes[0].set_ylabel("blur-circle diameter (px)")
    axes[0].set_ylim(0, None)
    fig.suptitle(f"{args.lens_mm:g} mm f/{args.fstop:g} (K = {k:.1f} px per dioptre), target {args.z_lo:g}-{args.z_hi:g} m: "
                 "dots = blur of an object on one focal plane, seen by the other camera", x=0.01, ha="left", color=INK, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
