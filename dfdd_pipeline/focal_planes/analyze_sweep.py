"""Figures and tables for one focal-plane sweep (``sweep_focal_planes.py`` output).

Usage: python dfdd_pipeline/focal_planes/analyze_sweep.py artifacts/focal_planes/sweeps/<name> [--sparsity 0.0]
Writes ``alpha_curves.png``, ``error_vs_depth.png`` and ``summary.md`` into the sweep folder.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill): categorical slots in fixed order, ink and chrome tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def load(sweep: Path):
    meta = json.loads((sweep / "meta.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (sweep / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return meta, records


def averaged(records: list[dict], sparsity: str) -> list[dict]:
    """Average seeds of the same pair."""
    groups = defaultdict(list)
    for r in records:
        groups[(round(r["near_u"], 6), round(r["far_u"], 6))].append(r)
    out = []
    for group in groups.values():
        first = dict(group[0])
        for field in ("mean_rel", "max_rel", "coverage", "mae_m"):
            first[field] = float(np.mean([g["test"][sparsity][field] for g in group]))
        first["mean_rel_sd"] = float(np.std([g["test"][sparsity]["mean_rel"] for g in group]))
        first["wide_rel"] = np.mean([np.array(g["wide"][sparsity]["per_depth_mae_m"]) / np.array(g["wide"]["depths_m"]) for g in group], axis=0)
        first["seeds"] = len(group)
        out.append(first)
    return out


def pair_label(r: dict) -> str:
    return f"{r['near_m']:.2f} / {r['far_m']:.2f} m"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--sparsity", default="0.0", help="0.0 (dense) or 0.9")
    args = parser.parse_args()
    meta, records = load(args.sweep)
    rows = averaged(records, args.sparsity)
    grid = [r for r in rows if not r.get("explicit")]
    explicit = [r for r in rows if r.get("explicit")]
    z_lo, z_hi = meta["z_lo"], meta["z_hi"]
    tag = "dense" if args.sparsity == "0.0" else f"{int(float(args.sparsity) * 100)}% sparse"
    suffix = "" if args.sparsity == "0.0" else f"_s{int(float(args.sparsity) * 100)}"
    best = min(grid, key=lambda r: r["mean_rel"])
    best_max = min(grid, key=lambda r: r["max_rel"])
    cam = meta["camera"]

    # ---- figure 1: error against focal-plane separation, one line per centre offset
    deltas = sorted({round(r["delta"], 4) for r in grid})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, field, title in ((axes[0], "mean_rel", "Mean error over the range"),
                             (axes[1], "max_rel", "Worst depth in the range")):
        style_axes(ax)
        for i, delta in enumerate(sorted(deltas, key=lambda d: (abs(d), d))):
            line = sorted([r for r in grid if round(r["delta"], 4) == delta], key=lambda r: r["alpha"])
            ax.plot([r["alpha"] for r in line], [100 * r[field] for r in line], color=SERIES[i], lw=2,
                    marker="o", ms=4.5, mec=SURFACE, mew=1.2, label=f"centre {delta:+.1f} W", solid_capstyle="round")
        for j, r in enumerate(explicit):  # explicit pairs: ink markers of different shapes, never a ninth hue
            ax.plot(r["alpha"], 100 * r[field], marker="DsP^v"[j % 5], ms=8, color=INK,
                    mec=SURFACE, mew=1.5, ls="none", label=f"pair {pair_label(r)}")
        ax.axvline(1.0, color=AXIS, lw=1)
        ax.text(1.0, ax.get_ylim()[1], " planes on range ends", color=MUTED, fontsize=8, va="top")
        ax.set_xlabel("focal-plane half-separation / range half-width (alpha, dioptres)")
        ax.set_ylabel(f"relative depth error, % ({tag})")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle(f"{cam['lens_mm']:g} mm f/{cam['fstop']:g}, target {z_lo:g}-{z_hi:g} m: best mean error at alpha = {best['alpha']:.2f}, "
                 f"centre {best['delta']:+.1f} W (focus {pair_label(best)})", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(args.sweep / f"alpha_curves{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 2: error against depth (wide sweep) for selected pairs
    ends = min((r for r in grid if abs(r["delta"]) < 1e-9), key=lambda r: abs(r["alpha"] - 1.0))
    chosen = [("best worst-depth error", best_max)]
    if abs(best["delta"] - best_max["delta"]) > 0.05 or abs(best["alpha"] - best_max["alpha"]) > 0.05:
        chosen.append(("best mean error", best))
    chosen.append(("planes on range ends", ends))
    near_ends = lambda r: abs(r["alpha"] - 1.0) < 0.06 and abs(r["delta"]) < 0.06  # noqa: E731
    chosen += [(f"pair {pair_label(r)}", r) for r in explicit if not near_ends(r)]
    chosen = chosen[:5]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axvspan(z_lo, z_hi, color="#2a78d6", alpha=0.07, lw=0)
    ax.axhline(10, color=AXIS, lw=1)
    ax.text(ax.get_xlim()[0], 10, " 10% working-range criterion", color=MUTED, fontsize=8, va="bottom")
    for i, (label, r) in enumerate(chosen):
        depths = np.array(records[0]["wide"]["depths_m"])
        ax.plot(depths, 100 * r["wide_rel"], color=SERIES[i], lw=2,
                label=f"{label}: focus {pair_label(r)} (alpha {r['alpha']:.2f}, centre {r['delta']:+.2f} W)")
        for plane in (r["near_m"], r["far_m"]):
            ax.plot(plane, 0, marker="|", ms=14, mew=2, color=SERIES[i], clip_on=False)
    ax.set_ylim(0, 25)
    ax.set_xlim(min(depths.min(), z_lo / 1.15), z_hi * 1.6)
    ax.set_xlabel("true depth (m); shaded = target range, ticks on the axis = focal planes")
    ax.set_ylabel(f"relative depth error, % ({tag})")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=INK2, loc="outside lower center", ncol=2)
    ax.set_title(f"Error against depth, calibrated on {z_lo:g}-{z_hi:g} m, scored on fresh slides", loc="left", color=INK, fontsize=11)
    fig.savefig(args.sweep / f"error_vs_depth{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure 3: response curves (median prediction against truth) on the wide sweep, best centre
    line = sorted([r for r in grid if abs(r["delta"] - best["delta"]) < 1e-9], key=lambda r: r["alpha"])
    picks = [r for r in line if any(abs(r["alpha"] - a) < 1e-6 for a in (0.3, 0.5, 0.7, 1.0, 1.5))] or line[:: max(1, len(line) // 5)]
    by_key = {(round(r["near_u"], 6), round(r["far_u"], 6)): r for r in records if r["seed"] == records[0]["seed"]}
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    depths = np.array(records[0]["wide"]["depths_m"])
    ax.axvspan(z_lo, z_hi, color=SERIES[0], alpha=0.07, lw=0)
    ax.plot(depths, depths, color=AXIS, lw=1)
    for i, r in enumerate(picks):
        raw = by_key[(round(r["near_u"], 6), round(r["far_u"], 6))]
        ax.plot(depths, raw["wide"][args.sparsity]["per_depth_median_m"], color=SERIES[i], lw=2,
                label=f"alpha {r['alpha']:.2f}: focus {pair_label(r)}")
        for plane in (r["near_m"], r["far_m"]):
            ax.plot(plane, depths.min(), marker="|", ms=12, mew=2, color=SERIES[i], clip_on=False)
    ax.set_xlim(depths.min(), depths.max())
    ax.set_ylim(depths.min(), depths.max())
    ax.set_xlabel("true depth (m); shaded = calibrated range, ticks = focal planes")
    ax.set_ylabel(f"median predicted depth (m, {tag})")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    ax.set_title(f"Depth response beyond the calibrated range (centre {best['delta']:+.1f} W)", loc="left", color=INK, fontsize=11)
    fig.savefig(args.sweep / f"response_curves{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- table
    lines = [f"# Sweep {args.sweep.name} ({tag})", "",
             f"Camera {cam['lens_mm']:g} mm f/{cam['fstop']:g}, K = {meta['k_px_per_diopter']:.1f} px/D; target {z_lo:g}-{z_hi:g} m "
             f"(centre {meta['c0_diopter']:.4f} D = {1 / meta['c0_diopter']:.2f} m, half-width W = {meta['w_diopter']:.4f} D, "
             f"K*W = {meta['range_blur_span_px']:.2f} px); noise {meta['noise']}.", "",
             "| alpha | centre (W) | near / far focus (m) | half-sep blur (px) | mean rel % | worst rel % | coverage | MAE (cm) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["mean_rel"]):
        lines.append(f"| {r['alpha']:.2f} | {r['delta']:+.2f} | {r['near_m']:.3f} / {r['far_m']:.3f}{' (explicit)' if r.get('explicit') else ''} | "
                     f"{r['b_half_px']:.2f} | {100 * r['mean_rel']:.2f} | {100 * r['max_rel']:.1f} | {r['coverage']:.2f} | {100 * r['mae_m']:.1f} |")
    (args.sweep / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"best mean: alpha {best['alpha']:.2f} delta {best['delta']:+.2f} focus {pair_label(best)} -> {100 * best['mean_rel']:.2f}% "
          f"(worst {100 * best['max_rel']:.1f}%)")
    print(f"best worst-depth: alpha {best_max['alpha']:.2f} delta {best_max['delta']:+.2f} focus {pair_label(best_max)} -> "
          f"worst {100 * best_max['max_rel']:.1f}% (mean {100 * best_max['mean_rel']:.2f}%)")
    for r in explicit:
        print(f"explicit {pair_label(r)}: alpha {r['alpha']:.2f} delta {r['delta']:+.2f} -> mean {100 * r['mean_rel']:.2f}% worst {100 * r['max_rel']:.1f}%")


if __name__ == "__main__":
    main()
