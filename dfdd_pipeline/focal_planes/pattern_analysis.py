"""Collect the best focal-plane separation of every sweep and look for the rule behind it.

For each sweep folder and centre offset, the best alpha (half-separation / range half-width, in
dioptres) is refined with a parabola through the three grid points around the minimum of the mean
relative error. It is reported with the camera's blur scale K (px per dioptre) and the range's
blur span K*W (px): if the best design depends on the optics only through blur in pixels, every
scenario with the same K*W (and noise level) shares the same best alpha.

Usage: python dfdd_pipeline/focal_planes/pattern_analysis.py [--sweeps artifacts/focal_planes/sweeps] [--prefix ap_ rg_]
Writes pattern.json, pattern.md and pattern.png into the sweeps folder.
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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.analyze_sweep import AXIS, INK, INK2, MUTED, SERIES, SURFACE, style_axes  # noqa: E402

RULE_BLUR_PX = 2.5  # best half-separation in blur pixels found across the aperture sweeps


def plain_log_axis(ax, ticks) -> None:
    """Log x axis labelled with plain numbers."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def refine_minimum(alphas: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Parabolic refinement (in log alpha) of the grid minimum; returns (alpha*, value*)."""
    i = int(np.argmin(values))
    if i == 0 or i == len(alphas) - 1:
        return float(alphas[i]), float(values[i])
    x = np.log(alphas[i - 1:i + 2])
    y = values[i - 1:i + 2]
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return float(alphas[i]), float(values[i])
    xm = -b / (2 * a)
    return float(np.exp(xm)), float(np.polyval([a, b, c], xm))


def tolerance_band(alphas: np.ndarray, values: np.ndarray, factor: float = 1.10) -> tuple[float, float]:
    """Range of alpha whose error is within ``factor`` of the minimum (interpolated on the grid)."""
    best = values.min()
    ok = values <= factor * best
    idx = np.nonzero(ok)[0]
    return float(alphas[idx.min()]), float(alphas[idx.max()])


def analyse(sweep: Path, sparsity: str = "0.0") -> list[dict]:
    meta = json.loads((sweep / "meta.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (sweep / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("explicit"):
            continue
        groups[round(r["delta"], 4)][round(r["alpha"], 4)].append(r)
    out = []
    k, w = meta["k_px_per_diopter"], meta["w_diopter"]
    for delta, by_alpha in groups.items():
        alphas = np.array(sorted(by_alpha))
        if len(alphas) < 3:
            continue
        mean_rel = np.array([np.mean([g["test"][sparsity]["mean_rel"] for g in by_alpha[a]]) for a in alphas])
        max_rel = np.array([np.mean([g["test"][sparsity]["max_rel"] for g in by_alpha[a]]) for a in alphas])
        a_star, e_star = refine_minimum(alphas, mean_rel)
        a_worst, e_worst = refine_minimum(alphas, max_rel)
        lo, hi = tolerance_band(alphas, mean_rel)
        ends = mean_rel[int(np.argmin(np.abs(alphas - 1.0)))]
        out.append({
            "sweep": sweep.name, "delta": delta, "k_px_per_diopter": k, "w_diopter": w, "kw_px": k * w,
            "z_lo": meta["z_lo"], "z_hi": meta["z_hi"], "ratio": meta["z_hi"] / meta["z_lo"],
            "fstop": meta["camera"]["fstop"], "lens_mm": meta["camera"]["lens_mm"],
            "noise": meta["noise"]["sigma"], "tone": meta.get("tone", "agx"), "psf": meta.get("psf", "disk"),
            "alpha_star": a_star, "mean_rel_star": e_star, "alpha_band_10pct": [lo, hi],
            "alpha_star_worst": a_worst, "max_rel_star": e_worst, "mean_rel_at_ends": float(ends),
            "center_blur_star_px": a_star * k * w, "end_blur_star_px": (1 + a_star) * k * w,
            "grid_alphas": alphas.tolist(), "grid_mean_rel": mean_rel.tolist(), "grid_max_rel": max_rel.tolist(),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=Path, default=ROOT / "artifacts/focal_planes/sweeps")
    parser.add_argument("--prefix", nargs="*", default=["ap_", "rg_", "nz_", "tn_", "ab_"])
    parser.add_argument("--sparsity", default="0.0")
    parser.add_argument("--delta", type=float, default=-0.2, help="Centre offset (units of W) whose lines are plotted")
    args = parser.parse_args()
    rows = []
    for sweep in sorted(p for p in args.sweeps.iterdir() if p.is_dir() and (p / "results.jsonl").exists()):
        if args.prefix and not any(sweep.name.startswith(pre) for pre in args.prefix):
            continue
        rows.extend(analyse(sweep, args.sparsity))
    suffix = "" if args.sparsity == "0.0" else f"_s{int(float(args.sparsity) * 100)}"
    (args.sweeps / f"pattern{suffix}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    lines = ["| sweep | centre | f/ | range (m) | K (px/D) | K*W (px) | best alpha | 10% band | best mean % | at ends % | best-worst alpha | centre blur (px) | end blur (px) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["sweep"][:3], r["kw_px"], r["delta"])):
        lines.append(f"| {r['sweep']} | {r['delta']:+.1f} | {r['fstop']:g} | {r['z_lo']:g}-{r['z_hi']:g} | {r['k_px_per_diopter']:.1f} | "
                     f"{r['kw_px']:.2f} | {r['alpha_star']:.2f} | {r['alpha_band_10pct'][0]:.2f}-{r['alpha_band_10pct'][1]:.2f} | "
                     f"{100 * r['mean_rel_star']:.2f} | {100 * r['mean_rel_at_ends']:.2f} | {r['alpha_star_worst']:.2f} | "
                     f"{r['center_blur_star_px']:.2f} | {r['end_blur_star_px']:.2f} |")
    (args.sweeps / f"pattern{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Figure: best alpha and best centre blur against the range's blur span, by scenario family.
    families = [("ap_", "aperture f/1.4-f/8, range 2-4.3 m"), ("rg_", "f/2, other ranges"), ("nz_", "f/2, 2-4.3 m, sensor noise")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)
    for i, (prefix, label) in enumerate(families):
        fam = [r for r in rows if r["sweep"].startswith(prefix) and abs(r["delta"] - args.delta) < 1e-9]
        if not fam:
            continue
        fam.sort(key=lambda r: r["kw_px"])
        x = [r["kw_px"] for r in fam]
        axes[0].plot(x, [r["alpha_star"] for r in fam], marker="o", ms=6, lw=0 if prefix != "ap_" else 2,
                     color=SERIES[i], mec=SURFACE, mew=1.5, label=label)
        axes[1].plot(x, [r["center_blur_star_px"] for r in fam], marker="o", ms=6, lw=0 if prefix != "ap_" else 2,
                     color=SERIES[i], mec=SURFACE, mew=1.5, label=label)
    kw = np.linspace(0.8, max([r["kw_px"] for r in rows] + [6.5]), 200)
    axes[0].plot(kw, RULE_BLUR_PX / kw, color=MUTED, lw=1, label=f"rule: K*h = {RULE_BLUR_PX:g} px")
    axes[1].axhline(RULE_BLUR_PX, color=MUTED, lw=1)
    axes[1].text(kw[-1], RULE_BLUR_PX, f"rule {RULE_BLUR_PX:g} px ", color=MUTED, fontsize=8, va="bottom", ha="right")
    axes[1].set_ylim(0, 6)
    axes[0].set_ylim(0, None)
    axes[0].axhline(1.0, color=AXIS, lw=1)
    axes[0].text(kw[-1], 1.0, "planes on range ends ", color=MUTED, fontsize=8, va="bottom", ha="right")
    axes[0].set_xlabel("range blur span K*W (px): blur of the range ends seen from its dioptric centre")
    axes[0].set_ylabel("best alpha = half-separation / range half-width")
    axes[0].set_title("Best focal-plane separation, relative to the range", loc="left", color=INK, fontsize=11)
    axes[1].set_xlabel("range blur span K*W (px)")
    axes[1].set_ylabel("best half-separation in blur pixels, K*h (px)")
    axes[1].set_title("The same optimum, in pixels of blur", loc="left", color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.savefig(args.sweeps / f"pattern{suffix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", args.sweeps / f"pattern{suffix}.png")

    # Noise: the same camera and range with independent sensor noise added to each image.
    noise_rows = sorted([r for r in rows if (r["sweep"].startswith("nz_") or r["sweep"] == "ap_f2_2to4p3")
                         and abs(r["delta"] - args.delta) < 1e-9], key=lambda r: r["noise"])
    if len(noise_rows) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            style_axes(ax)
        for i, r in enumerate(noise_rows):
            axes[0].plot(r["grid_alphas"], 100 * np.array(r["grid_mean_rel"]), color=SERIES[i], lw=2, marker="o", ms=4,
                         mec=SURFACE, mew=1, label=f"noise {r['noise']:g} grey levels")
        axes[0].axvline(1.0, color=AXIS, lw=1)
        axes[0].set_xlabel("alpha = half-separation / range half-width")
        axes[0].set_ylabel("mean relative error over the range, %")
        axes[0].set_title("Error against separation", loc="left", color=INK, fontsize=11)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
        sig = [max(r["noise"], 0.05) for r in noise_rows]
        axes[1].plot(sig, [r["center_blur_star_px"] for r in noise_rows], color=SERIES[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
        axes[1].axhline(RULE_BLUR_PX, color=MUTED, lw=1)
        plain_log_axis(axes[1], [0.05, 0.1, 0.25, 0.5, 1, 2])
        axes[1].set_ylim(0, 6)
        axes[1].set_xlabel("noise per image, grey levels (log)")
        axes[1].set_ylabel("best half-separation K*h (px)")
        axes[1].set_title("Best separation grows with noise", loc="left", color=INK, fontsize=11)
        fig.suptitle(f"85 mm f/2, target 2-4.3 m, pair centre {args.delta:+.1f} W: noise widens and flattens the optimum",
                     x=0.01, ha="left", color=INK, fontsize=12)
        out = args.sweeps / f"noise{suffix}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", out)

    # Ablations: switch off one error mechanism at a time (85 mm f/2, 2-4.3 m, centred pairs).
    ablations = [("ap_f2_2to4p3", "pipeline as rendered (AgX, render noise, disk blur)"),
                 ("tn_f2_2to4p3_linear", "linear sensor"),
                 ("tn_f2_2to4p3_linear_clean", "linear sensor, no noise or quantisation"),
                 ("ab_f2_2to4p3_gauss", "Gaussian blur instead of the aperture disk"),
                 ("ab_f2_2to4p3_gauss_linear_clean", "Gaussian blur, linear, no noise"),
                 ("ab_f2_2to4p3_3scales", "three pyramid scales instead of two"),
                 ("ab_f4_1to2p15_invariance", "f/4 on 1.0-2.15 m (same K*W and ratio)")]
    # the ablation sweeps use centred pairs only
    present = [(name, label, next((r for r in rows if r["sweep"] == name and abs(r["delta"]) < 1e-9), None)) for name, label in ablations]
    present = [(n, l, r) for n, l, r in present if r is not None]
    if len(present) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            style_axes(ax)
        for i, (name, label, r) in enumerate(present):
            x = np.array(r["grid_alphas"]) * r["kw_px"]
            err = 100 * np.array(r["grid_mean_rel"])
            axes[0].plot(x, err, color=SERIES[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1,
                         label=f"{label}: best K*h {r['center_blur_star_px']:.1f} px, {100 * r['mean_rel_star']:.2f} %")
            axes[1].plot(x, err / err.min(), color=SERIES[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
        for ax in axes:
            plain_log_axis(ax, [0.5, 1, 1.5, 2, 3, 5, 7, 10])
            ax.set_xlabel("half-separation in blur pixels, K*h (log)")
        axes[0].set_ylabel("mean relative error over the range, %")
        axes[0].set_title("Error", loc="left", color=INK, fontsize=11)
        axes[1].set_ylim(0.95, 2.5)
        axes[1].set_ylabel("error / best error of that variant")
        axes[1].set_title("Error relative to each variant's best", loc="left", color=INK, fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=INK2, loc="outside lower center", ncol=2)
        fig.suptitle("Which mechanism sets the best separation (85 mm f/2, 2-4.3 m, centred pairs)", x=0.01, ha="left", color=INK, fontsize=12)
        out = args.sweeps / f"ablations{suffix}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", out)

    # Collapse: each scenario's error curve against the half-separation in blur pixels (K*h).
    for prefix, title in (("ap_", "85 mm f/1.4 to f/8, target 2-4.3 m"), ("rg_", "85 mm f/2, different target ranges")):
        fam = sorted([r for r in rows if r["sweep"].startswith(prefix) and abs(r["delta"] - args.delta) < 1e-9], key=lambda r: r["kw_px"])
        if not fam:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            style_axes(ax)
        for i, r in enumerate(fam[: len(SERIES)]):
            alphas, err = np.array(r["grid_alphas"]), 100 * np.array(r["grid_mean_rel"])
            label = (f"f/{r['fstop']:g}" if prefix == "ap_" else f"{r['z_lo']:g}-{r['z_hi']:g} m") + f" (K*W {r['kw_px']:.1f} px)"
            axes[0].plot(alphas, err, color=SERIES[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1, label=label)
            axes[1].plot(alphas * r["kw_px"], err / err.min(), color=SERIES[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
        plain_log_axis(axes[0], [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5])
        axes[0].set_xlabel("alpha = half-separation / range half-width (log)")
        axes[0].set_ylabel("mean relative error over the range, %")
        axes[0].set_title("Against separation relative to the range", loc="left", color=INK, fontsize=11)
        plain_log_axis(axes[1], [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15])
        axes[1].set_ylim(0.95, 2.2)
        axes[1].set_xlabel("half-separation in blur pixels, K*h (log)")
        axes[1].set_ylabel("error / best error of that scenario")
        axes[1].set_title("Against separation in pixels of blur", loc="left", color=INK, fontsize=11)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
        fig.suptitle(title + f" (pair centre {args.delta:+.1f} W from the range's dioptric centre)", x=0.01, ha="left", color=INK, fontsize=12)
        out = args.sweeps / f"collapse_{prefix.rstrip('_')}{suffix}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
