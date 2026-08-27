"""Produce robust working-range metrics and presentation-ready comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPARSITIES = (0.0, 0.5, 0.8, 0.9, 0.95)


def confidence_mask(conf: np.ndarray, sparsity: float) -> np.ndarray:
    flat = conf.reshape(conf.shape[0], -1)
    threshold = np.quantile(flat, sparsity, axis=1)[:, None, None, None]
    return conf >= threshold


def per_depth_metrics(z_true: np.ndarray, z_pred: np.ndarray, conf: np.ndarray, sparsity: float):
    mask = confidence_mask(conf, sparsity)
    error = np.abs(z_pred - z_true)
    axes = tuple(range(1, error.ndim))
    count = mask.sum(axis=axes)
    mae = (error * mask).sum(axis=axes) / np.maximum(count, 1)
    truth = z_true.mean(axis=axes)
    return truth, mae, count


def longest_working_interval(depth: np.ndarray, mae: np.ndarray):
    valid = mae < 0.1 * depth
    best = None
    start = None
    for idx, is_valid in enumerate(np.r_[valid, False]):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            end = idx - 1
            candidate = (depth[start], depth[end], end - start + 1)
            if best is None or candidate[2] > best[2]:
                best = candidate
            start = None
    return None if best is None else (float(best[0]), float(best[1]))


def load_arrays(result_dir: Path):
    return tuple(np.load(result_dir / name) for name in ("z_true.npy", "z_pred.npy", "conf.npy"))


def analyze_one(result_dir: Path, output_dir: Path, label: str) -> dict:
    z_true, z_pred, conf = load_arrays(result_dir)
    border = 11
    z_true = z_true[..., border:-border, border:-border]
    z_pred = z_pred[..., border:-border, border:-border]
    conf = conf[..., border:-border, border:-border]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"label": label, "result_dir": str(result_dir), "sparsities": {}}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for sparsity in SPARSITIES:
        depth, mae, count = per_depth_metrics(z_true, z_pred, conf, sparsity)
        wr = longest_working_interval(depth, mae)
        summary["sparsities"][str(sparsity)] = {
            "mae_m": float(np.mean(mae)),
            "working_range_m": wr,
            "working_range_width_m": 0.0 if wr is None else wr[1] - wr[0],
            "kept_fraction": float(np.mean(count) / np.prod(z_true.shape[1:])),
        }
        ax.plot(depth, mae, marker="o", label=f"{int(sparsity * 100)}% sparse")
    ax.plot(depth, 0.1 * depth, "k--", linewidth=2, label="10% relative-error threshold")
    ax.set(xlabel="True depth (m)", ylabel="Mean absolute error (m)", title=f"Error vs. true depth — {label}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_true_depth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    sparsity_grid = np.linspace(0.0, 0.99, 100)
    errors = []
    for sparsity in sparsity_grid:
        _, mae, _ = per_depth_metrics(z_true, z_pred, conf, sparsity)
        errors.append(np.mean(mae))
    ax.plot(sparsity_grid * 100, errors, linewidth=2.5)
    ax.set(xlabel="Sparsity (%)", ylabel="Mean absolute error (m)", title=f"Accuracy–density trade-off — {label}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_sparsity.png", dpi=180)
    plt.close(fig)

    for sparsity in (0.0, 0.9):
        mask = confidence_mask(conf, sparsity)
        truth = z_true[mask]
        prediction = z_pred[mask]
        limits = (float(z_true.min()), float(z_true.max()))
        fig, ax = plt.subplots(figsize=(6.4, 5.8))
        hist = ax.hist2d(truth, prediction, bins=max(8, z_true.shape[0]), range=(limits, limits), cmap="magma")
        ax.plot(limits, limits, "w--", linewidth=2)
        ax.set(xlabel="True depth (m)", ylabel="Predicted depth (m)", title=f"Depth heatmap — {label} — {int(sparsity*100)}% sparse")
        fig.colorbar(hist[3], ax=ax, label="Pixel count")
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_sparsity_{int(sparsity*100):02d}.png", dpi=180)
        plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def comparison_plot(summaries: list[dict], output_dir: Path) -> None:
    labels = [item["label"] for item in summaries]
    maes = [item["sparsities"]["0.9"]["mae_m"] for item in summaries]
    widths = [item["sparsities"]["0.9"]["working_range_width_m"] for item in summaries]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(x, maes, color="#6f55b5")
    axes[0].set(ylabel="MAE (m)", title="90% sparsity: lower is better")
    axes[1].bar(x, widths, color="#2d8f85")
    axes[1].set(ylabel="Working-range width (m)", title="90% sparsity: higher is better")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    with (output_dir / "ablation_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("configuration", "mae_m_at_90pct_sparsity", "working_range_width_m", "working_range_m"))
        for item in summaries:
            metric = item["sparsities"]["0.9"]
            writer.writerow((item["label"], metric["mae_m"], metric["working_range_width_m"], metric["working_range_m"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, help="LABEL=path containing z_true.npy/z_pred.npy/conf.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for specification in args.result:
        label, path = specification.split("=", 1)
        summaries.append(analyze_one(Path(path), args.output_dir / label, label))
    comparison_plot(summaries, args.output_dir)


if __name__ == "__main__":
    main()
