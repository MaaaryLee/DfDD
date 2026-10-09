"""Figure for ``blender_check.py``: per-depth relative error of each focal pair, Blender vs simulator.

Usage: python dfdd_pipeline/focal_planes/plot_blender_check.py artifacts/focal_planes/blender/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.focal_planes.analyze_sweep import AXIS, INK, INK2, MUTED, SERIES, SURFACE, style_axes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    results = json.loads((args.folder / "results.json").read_text(encoding="utf-8"))
    results.sort(key=lambda r: np.mean(r["blender"]["0.0"]["per_depth_mae_m"]) / np.mean(r["depths_m"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, sparsity, title in ((axes[0], "0.0", "Dense (all pixels)"), (axes[1], "0.9", "90% sparse (top 10% confidence)")):
        style_axes(ax)
        for i, r in enumerate(results):
            depths = np.array(r["depths_m"])
            blender = 100 * np.array(r["blender"][sparsity]["per_depth_mae_m"]) / depths
            sim = 100 * np.array(r["sim"][sparsity]["per_depth_mae_m"]) / depths
            label = (f"focus {r['near_m']:.2f} / {r['far_m']:.2f} m: Blender {100 * r['blender'][sparsity]['mean_rel']:.2f}%, "
                     f"sim {100 * r['sim'][sparsity]['mean_rel']:.2f}%")
            ax.plot(depths, blender, color=SERIES[i], lw=2, label=label)
            ax.plot(depths, sim, color=SERIES[i], lw=1, alpha=0.9, ls=(0, (1, 1.5)))
            for plane in (r["near_m"], r["far_m"]):
                if depths.min() <= plane <= depths.max():
                    ax.plot(plane, 0, marker="|", ms=14, mew=2, color=SERIES[i], clip_on=False)
        ax.set_ylim(0, None)
        ax.set_xlabel("true depth (m); ticks on the axis = focal planes")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", title="solid = Blender, dotted = simulator",
                  title_fontsize=8)
    axes[0].set_ylabel("relative depth error (MAE / depth), %")
    best = results[0]
    gap = max(abs(r["blender"]["0.0"]["mean_rel"] - r["sim"]["0.0"]["mean_rel"]) for r in results)
    fig.suptitle(f"Best tested pair on Blender slides: focus {best['near_m']:.2f} / {best['far_m']:.2f} m "
                 f"({100 * best['blender']['0.0']['mean_rel']:.2f}% dense); simulator within {100 * gap:.2f} points of Blender",
                 x=0.01, ha="left", color=INK, fontsize=12)
    out = args.folder / "blender_vs_sim.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)
    for r in results:
        print(f"{r['near_m']:.3f}/{r['far_m']:.3f}: Blender dense {100 * r['blender']['0.0']['mean_rel']:.2f}% worst {100 * r['blender']['0.0']['max_rel']:.1f}% "
              f"| 90% {100 * r['blender']['0.9']['mean_rel']:.2f}% worst {100 * r['blender']['0.9']['max_rel']:.1f}% "
              f"|| sim dense {100 * r['sim']['0.0']['mean_rel']:.2f}% | 90% {100 * r['sim']['0.9']['mean_rel']:.2f}%")


if __name__ == "__main__":
    main()
