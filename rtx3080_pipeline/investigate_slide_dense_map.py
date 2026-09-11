"""Check whether a linear-slide dense map carries checkerboard texture instead of depth.

The slide target is a fronto-parallel plane, so the true depth map is constant across
every frame. Any spatial structure in the prediction is therefore texture leakage, and
this script quantifies how much of it tracks the checker edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


BORDER = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Rendered slide frames plus metadata.json")
    parser.add_argument("--result-dir", type=Path, required=True, help="Contains z_true.npy, z_pred.npy, conf.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-depth", type=float, default=1.0)
    parser.add_argument("--edge-quantile", type=float, default=0.9)
    return parser.parse_args()


def crop(array: np.ndarray) -> np.ndarray:
    return array[..., BORDER:-BORDER, BORDER:-BORDER]


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def build_figure(
    output_path: Path,
    near: np.ndarray,
    far: np.ndarray,
    z_pred: np.ndarray,
    conf: np.ndarray,
    error: np.ndarray,
    grad: np.ndarray,
    true_depth: float,
    row: int,
) -> None:
    fig = plt.figure(figsize=(14.4, 10.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.75))

    def panel(position, image, title, cmap, vmin=None, vmax=None, label=None):
        ax = fig.add_subplot(grid[position])
        handle = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axhline(row, color="cyan", linewidth=0.9, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(handle, ax=ax, shrink=0.8, label=label)
        return ax

    panel((0, 0), near, "Near-focus input", "gray", label="Intensity")
    panel((0, 1), far, "Far-focus input", "gray", label="Intensity")
    panel((0, 2), grad, "Image gradient magnitude", "magma", label="|grad I|")

    span = max(0.05, 3.0 * float(np.std(z_pred)))
    panel(
        (1, 0),
        z_pred,
        f"Dense prediction (true = {true_depth:.3f} m)",
        "viridis",
        vmin=true_depth - span,
        vmax=true_depth + span,
        label="Depth (m)",
    )
    panel((1, 1), conf, "Confidence", "cividis", label="Confidence")
    panel((1, 2), error, "Absolute error", "inferno", vmin=0.0, vmax=float(np.percentile(error, 98)), label="Error (m)")

    ax = fig.add_subplot(grid[2, :])
    columns = np.arange(z_pred.shape[1])
    ax.plot(columns, z_pred[row], color="#2f6fb5", linewidth=1.4, label="Predicted depth")
    ax.axhline(true_depth, color="black", linestyle="--", linewidth=1.6, label=f"True depth {true_depth:.3f} m")
    ax.set_xlabel("Image column")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim(true_depth - span, true_depth + span)
    ax.grid(alpha=0.25)

    intensity_axis = ax.twinx()
    intensity_axis.plot(columns, near[row], color="#c2704a", linewidth=1.0, alpha=0.75, label="Near-focus intensity")
    intensity_axis.set_ylabel("Intensity")

    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = intensity_axis.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, loc="upper right", fontsize=9, ncol=3)
    ax.set_title(f"Scanline through image row {row}: does predicted depth oscillate with the checker?", fontsize=11)

    fig.suptitle("Linear-slide dense map: depth or checker texture?", fontsize=14, fontweight="semibold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    depths = np.asarray(json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))["depths_m"], dtype=float)
    z_true = crop(np.load(args.result_dir / "z_true.npy"))
    z_pred = crop(np.load(args.result_dir / "z_pred.npy"))
    conf = crop(np.load(args.result_dir / "conf.npy"))
    if z_true.shape[0] != depths.size:
        raise ValueError(f"{z_true.shape[0]} calibration frames but {depths.size} rendered depths")

    index = int(np.argmin(np.abs(depths - args.target_depth)))
    true_depth = float(depths[index])
    near = crop(load_gray(args.data_dir / f"cam_0_500_480_{index}.png"))
    far = crop(load_gray(args.data_dir / f"cam_1_500_480_{index}.png"))
    prediction = np.squeeze(z_pred[index])
    confidence = np.squeeze(conf[index])
    error = np.abs(prediction - true_depth)
    grad = gradient_magnitude(0.5 * (near + far))

    edge = grad >= np.quantile(grad, args.edge_quantile)
    flat = grad <= np.quantile(grad, 1.0 - args.edge_quantile)

    metrics = {
        "data_dir": str(args.data_dir),
        "result_dir": str(args.result_dir),
        "frame_index": index,
        "true_depth_m": true_depth,
        "true_depth_spatial_std_m": float(np.std(np.squeeze(z_true[index]))),
        "prediction_spatial_std_m": float(np.std(prediction)),
        "prediction_median_m": float(np.median(prediction)),
        "mae_m": float(np.mean(error)),
        "mae_on_checker_edges_m": float(np.mean(error[edge])),
        "mae_on_flat_squares_m": float(np.mean(error[flat])),
        "corr_image_gradient_vs_confidence": correlation(grad, confidence),
        "corr_image_gradient_vs_abs_error": correlation(grad, error),
        "fraction_within_1cm": float(np.mean(error <= 0.01)),
        "fraction_within_5cm": float(np.mean(error <= 0.05)),
        "interpretation": (
            "The slide target is a flat plane, so true depth has zero spatial variance. A trustworthy dense map is "
            "flat too. Confidence is expected to correlate with image gradient because DfDD needs texture, but if "
            "the prediction itself oscillates with the checker, the dense map is imaging texture rather than depth."
        ),
    }

    build_figure(
        args.output_dir / "slide_dense_map_check.png",
        near,
        far,
        prediction,
        confidence,
        error,
        grad,
        true_depth,
        row=prediction.shape[0] // 2,
    )
    (args.output_dir / "slide_dense_map_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
