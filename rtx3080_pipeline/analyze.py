"""Produce robust working-range metrics and presentation-ready comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np


SPARSITIES = (0.0, 0.5, 0.8, 0.9, 0.95)


def sparsity_label(sparsity: float) -> str:
    if sparsity == 0.0:
        return "dense (100% pixels)"
    return f"{int(sparsity * 100)}% sparse ({int(round((1.0 - sparsity) * 100))}% kept)"


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


def response_slope(z_true: np.ndarray, z_pred: np.ndarray, conf: np.ndarray, sparsity: float) -> float:
    """Slope of per-frame predicted median vs true depth.

    A collapsed fit predicts a near-constant depth, which still crosses the truth line
    somewhere and fakes a working range. Only the slope distinguishes that from a real
    depth response, so it gates every other number in the summary.
    """
    mask = confidence_mask(conf, sparsity)
    axes = tuple(range(1, z_true.ndim))
    truth = z_true.mean(axis=axes)
    median = np.array(
        [np.median(z_pred[i][mask[i]]) if mask[i].any() else np.nan for i in range(z_true.shape[0])]
    )
    finite = np.isfinite(median)
    if finite.sum() < 2:
        return float("nan")
    return float(np.polyfit(truth[finite], median[finite], 1)[0])


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


def representative_indices(count: int, maximum: int = 6) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, min(count, maximum), dtype=int))


def depth_image(array: np.ndarray, index: int) -> np.ndarray:
    image = np.squeeze(array[index])
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D depth image after squeezing, got shape {image.shape}")
    return image


def plot_dense_depth_grid(z_true: np.ndarray, z_pred: np.ndarray, output_dir: Path, label: str) -> None:
    depth = z_true.mean(axis=tuple(range(1, z_true.ndim)))
    indices = representative_indices(z_true.shape[0])
    error = np.abs(z_pred - z_true)
    error_vmax = float(np.percentile(error, 99))
    if not np.isfinite(error_vmax) or error_vmax <= 0:
        error_vmax = float(np.max(error)) if float(np.max(error)) > 0 else 1.0

    fig, axes = plt.subplots(3, len(indices), figsize=(2.35 * len(indices), 7.4), squeeze=False, constrained_layout=True)
    depth_image_handle = None
    depth_norm = plt.Normalize(float(z_true.min()), float(z_true.max()))
    row_labels = ("True depth", "Dense prediction", "Dense abs. error")

    for col, idx in enumerate(indices):
        true_map = depth_image(z_true, int(idx))
        pred_map = depth_image(z_pred, int(idx))
        error_map = np.abs(pred_map - true_map)

        # Prediction and error panels are scaled per panel so that residual checker
        # structure stays visible where the prediction is accurate and its spatial
        # spread is small. Each panel prints its own range: without that, per-panel
        # scaling makes an accurate panel look as bad as an inaccurate one.
        pred_lo, pred_hi = (float(v) for v in np.percentile(pred_map, [1, 99]))
        if pred_hi <= pred_lo:
            pred_lo, pred_hi = pred_lo - 0.005, pred_lo + 0.005
        error_hi = float(np.percentile(error_map, 99)) or 1e-3

        maps = (true_map, pred_map, error_map)
        cmaps = ("viridis", "viridis", "inferno")
        norms = (depth_norm, plt.Normalize(pred_lo, pred_hi), plt.Normalize(0.0, error_hi))
        footers = (
            None,
            f"{pred_lo:.3f}–{pred_hi:.3f} m\nσ = {float(np.std(pred_map)) * 100:.1f} cm",
            f"0–{error_hi * 100:.1f} cm\nMAE = {float(np.mean(error_map)) * 100:.1f} cm",
        )
        for row, (image, cmap, norm, footer) in enumerate(zip(maps, cmaps, norms, footers)):
            ax = axes[row, col]
            handle = ax.imshow(image, cmap=cmap, norm=norm)
            if row == 0:
                depth_image_handle = handle
                ax.set_title(f"{depth[int(idx)]:.2f} m", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if footer is not None:
                ax.set_xlabel(footer, fontsize=7.5, labelpad=2)
            if col == 0:
                ax.set_ylabel(row_labels[row], rotation=0, ha="right", va="center", labelpad=46, fontweight="semibold")

    fig.suptitle(
        f"Dense spatial depth maps — {label}\n"
        "prediction and error panels are scaled per panel; compare the printed ranges, not the colours",
        fontweight="semibold",
        fontsize=12,
    )
    if depth_image_handle is not None:
        fig.colorbar(depth_image_handle, ax=axes[0, :].ravel().tolist(), shrink=0.9, label="True depth (m)")
    fig.savefig(output_dir / "dense_depth_grid.png", dpi=180)
    plt.close(fig)


def set_centimeter_error_axis(ax, max_error_m: float) -> None:
    upper = max(0.03, float(max_error_m) * 1.08)
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(which="major", alpha=0.25)
    ax.grid(which="minor", alpha=0.12)


def heatmap_limits(truth: np.ndarray, prediction: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([truth[np.isfinite(truth)], prediction[np.isfinite(prediction)]])
    if values.size == 0:
        return fallback
    lo = min(float(fallback[0]), float(np.percentile(values, 0.5)))
    hi = max(float(fallback[1]), float(np.percentile(values, 99.5)))
    pad = max(0.01, 0.03 * (hi - lo))
    return lo - pad, hi + pad


def slide_position_edges(depth: np.ndarray) -> np.ndarray:
    """One heatmap column per slide position, so no column falls between two samples."""
    depth = np.unique(depth)
    step = float(np.median(np.diff(depth))) if depth.size > 1 else 0.01
    return np.append(depth - step / 2.0, depth[-1] + step / 2.0)


def analyze_one(result_dir: Path, output_dir: Path, label: str) -> dict:
    z_true, z_pred, conf = load_arrays(result_dir)
    border = 11
    z_true = z_true[..., border:-border, border:-border]
    z_pred = z_pred[..., border:-border, border:-border]
    conf = conf[..., border:-border, border:-border]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dense_depth_grid(z_true, z_pred, output_dir, label)

    summary = {"label": label, "result_dir": str(result_dir), "sparsities": {}}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    max_mae = 0.0
    for sparsity in SPARSITIES:
        depth, mae, count = per_depth_metrics(z_true, z_pred, conf, sparsity)
        max_mae = max(max_mae, float(np.nanmax(mae)))
        wr = longest_working_interval(depth, mae)
        summary["sparsities"][str(sparsity)] = {
            "mae_m": float(np.mean(mae)),
            "working_range_m": wr,
            "working_range_width_m": 0.0 if wr is None else wr[1] - wr[0],
            "kept_fraction": float(np.mean(count) / np.prod(z_true.shape[1:])),
            "slope": response_slope(z_true, z_pred, conf, sparsity),
        }
        ax.plot(depth, mae, marker="o", label=sparsity_label(sparsity))
    ax.plot(depth, 0.1 * depth, "k--", linewidth=2, label="10% relative-error threshold")
    ax.set(xlabel="True depth (m)", ylabel="Mean absolute error (m)", title=f"Error vs. true depth — {label}")
    set_centimeter_error_axis(ax, max(max_mae, float(np.nanmax(0.1 * depth))))
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
    set_centimeter_error_axis(ax, max(errors))
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_sparsity.png", dpi=180)
    plt.close(fig)

    for sparsity in (0.0, 0.9):
        mask = confidence_mask(conf, sparsity)
        truth = z_true[mask]
        prediction = z_pred[mask]
        truth_limits = (float(z_true.min()), float(z_true.max()))
        limits = heatmap_limits(truth, prediction, truth_limits)
        x_edges = slide_position_edges(z_true.mean(axis=tuple(range(1, z_true.ndim))))
        y_edges = np.arange(limits[0], limits[1] + 0.01, 0.01)
        counts, _, _ = np.histogram2d(truth, prediction, bins=[x_edges, y_edges])
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        # imshow rather than hist2d: both axes are uniform 1 cm bins, and pcolormesh
        # leaves antialiasing seams between quads that read as missing slide positions.
        image = ax.imshow(
            np.ma.masked_equal(counts.T, 0.0),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap="magma",
            norm=LogNorm(),
        )
        ax.plot(limits, limits, "--", color="#39ff14", linewidth=2, label="ideal y = x")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(limits)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="True depth (m)", ylabel="Predicted depth (m)", title=f"Depth heatmap — {label} — {sparsity_label(sparsity)}")
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.grid(which="minor", color="white", alpha=0.05, linewidth=0.4)
        fig.colorbar(image, ax=ax, label="Pixel count")
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_sparsity_{int(sparsity*100):02d}.png", dpi=180)
        if sparsity == 0.0:
            fig.savefig(output_dir / "heatmap_dense.png", dpi=180)
        plt.close(fig)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def comparison_plot(summaries: list[dict], output_dir: Path) -> None:
    labels = [item["label"] for item in summaries]
    dense_maes = [item["sparsities"]["0.0"]["mae_m"] for item in summaries]
    sparse_maes = [item["sparsities"]["0.9"]["mae_m"] for item in summaries]
    dense_widths = [item["sparsities"]["0.0"]["working_range_width_m"] for item in summaries]
    sparse_widths = [item["sparsities"]["0.9"]["working_range_width_m"] for item in summaries]
    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(x - width / 2, dense_maes, width, color="#3b82a0", label="dense")
    axes[0].bar(x + width / 2, sparse_maes, width, color="#6f55b5", label="90% sparse")
    axes[0].set(ylabel="MAE (m)", title="Dense vs sparse: lower is better")
    axes[1].bar(x - width / 2, dense_widths, width, color="#76a665", label="dense")
    axes[1].bar(x + width / 2, sparse_widths, width, color="#2d8f85", label="90% sparse")
    axes[1].set(ylabel="Working-range width (m)", title="Dense vs sparse: higher is better")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "configuration",
            "dense_mae_m",
            "dense_working_range_width_m",
            "dense_working_range_m",
            "mae_m_at_90pct_sparsity",
            "working_range_width_m_at_90pct_sparsity",
            "working_range_m_at_90pct_sparsity",
        ))
        for item in summaries:
            dense = item["sparsities"]["0.0"]
            sparse = item["sparsities"]["0.9"]
            writer.writerow((
                item["label"],
                dense["mae_m"],
                dense["working_range_width_m"],
                dense["working_range_m"],
                sparse["mae_m"],
                sparse["working_range_width_m"],
                sparse["working_range_m"],
            ))


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
