"""Diagnose whether a DfDD dense map follows true depth edges or RGB texture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-depth-min", type=float)
    parser.add_argument("--valid-depth-max", type=float)
    return parser.parse_args()


def normalize01(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = values.astype(np.float32).copy()
    valid = mask & np.isfinite(output)
    if not np.any(valid):
        return np.zeros_like(output, dtype=np.float32)
    lo = float(np.nanpercentile(output[valid], 2))
    hi = float(np.nanpercentile(output[valid], 98))
    output = (output - lo) / max(hi - lo, 1e-7)
    return np.clip(output, 0.0, 1.0)


def gradient_magnitude(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = values.astype(np.float32).copy()
    if np.any(valid):
        filled[~valid] = float(np.nanmedian(filled[valid]))
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def object_edge_mask(object_id: np.ndarray, valid: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(valid, dtype=bool)
    edges[:, 1:] |= object_id[:, 1:] != object_id[:, :-1]
    edges[1:, :] |= object_id[1:, :] != object_id[:-1, :]
    return edges & valid


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask = mask & np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def per_object_depth_table(
    object_id: np.ndarray,
    names_by_id: dict[str, str],
    true_depth: np.ndarray,
    dfdd_depth: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows = []
    for object_value in sorted(np.unique(object_id)):
        if object_value == 0:
            continue
        mask = valid & (object_id == object_value)
        if not np.any(mask):
            continue
        true_med = float(np.nanmedian(true_depth[mask]))
        pred_med = float(np.nanmedian(dfdd_depth[mask]))
        rows.append(
            {
                "object_id": int(object_value),
                "object_name": names_by_id.get(str(int(object_value)), f"object_{int(object_value)}"),
                "pixels": int(np.count_nonzero(mask)),
                "true_median_depth_m": true_med,
                "dfdd_median_depth_m": pred_med,
                "median_error_m": float(abs(pred_med - true_med)),
                "mae_m": float(np.nanmean(np.abs(dfdd_depth[mask] - true_depth[mask]))),
            }
        )
    return rows


def save_figure(
    output_path: Path,
    rgb: np.ndarray,
    true_depth: np.ndarray,
    dfdd_depth: np.ndarray,
    dfdd_error: np.ndarray,
    dfdd_grad: np.ndarray,
    true_edges: np.ndarray,
    rgb_texture_edges: np.ndarray,
    valid: np.ndarray,
) -> None:
    vmin = float(np.nanpercentile(true_depth[valid], 2))
    vmax = float(np.nanpercentile(true_depth[valid], 98))
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), constrained_layout=True)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB")

    depth_panel = axes[0, 1].imshow(np.ma.masked_where(~valid, true_depth), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].contour(true_edges.astype(float), levels=[0.5], colors="white", linewidths=0.7)
    axes[0, 1].set_title("True depth + object edges")
    fig.colorbar(depth_panel, ax=axes[0, 1], shrink=0.78, label="Depth (m)")

    dfdd_panel = axes[0, 2].imshow(np.ma.masked_where(~valid, dfdd_depth), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].contour(true_edges.astype(float), levels=[0.5], colors="white", linewidths=0.7)
    axes[0, 2].set_title("DfDD dense + true edges")
    fig.colorbar(dfdd_panel, ax=axes[0, 2], shrink=0.78, label="Depth (m)")

    grad_panel = axes[1, 0].imshow(normalize01(dfdd_grad, valid), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].contour(true_edges.astype(float), levels=[0.5], colors="cyan", linewidths=0.55)
    axes[1, 0].set_title("DfDD depth gradient")
    fig.colorbar(grad_panel, ax=axes[1, 0], shrink=0.78, label="Normalized gradient")

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(np.ma.masked_where(~rgb_texture_edges, rgb_texture_edges), cmap="cool", alpha=0.65)
    axes[1, 1].set_title("RGB checker texture edges")

    error_vmax = float(np.nanpercentile(dfdd_error[valid], 98))
    err_panel = axes[1, 2].imshow(np.ma.masked_where(~valid, dfdd_error), cmap="magma", vmin=0.0, vmax=max(0.02, error_vmax))
    axes[1, 2].contour(true_edges.astype(float), levels=[0.5], colors="white", linewidths=0.7)
    axes[1, 2].set_title("DfDD absolute error")
    fig.colorbar(err_panel, ax=axes[1, 2], shrink=0.78, label="Error (m)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Dense DfDD Map Edge/Texture Diagnostic", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(args.scene_dir / "rgb_reference.png").convert("RGB"))
    true_depth = np.load(args.scene_dir / "depth_true_m.npy")
    object_id = np.load(args.scene_dir / "object_id.npy")
    dfdd_depth = np.load(args.alignment_dir / "dfdd_depth_m.npy")
    metadata = json.loads((args.scene_dir / "metadata.json").read_text(encoding="utf-8"))
    names_by_id = {str(key): value for key, value in metadata.get("object_ids", {}).items()}

    valid_mask_path = args.alignment_dir / "valid_eval_mask.npy"
    if valid_mask_path.exists():
        valid = np.load(valid_mask_path).astype(bool)
    else:
        valid = np.isfinite(true_depth) & np.isfinite(dfdd_depth)
    interior = np.zeros_like(valid, dtype=bool)
    interior[11:-11, 11:-11] = True
    valid &= interior
    if args.valid_depth_min is not None:
        valid &= true_depth >= args.valid_depth_min
    if args.valid_depth_max is not None:
        valid &= true_depth <= args.valid_depth_max

    true_grad = gradient_magnitude(true_depth, valid)
    object_edges = object_edge_mask(object_id, valid)
    depth_edges = (true_grad > 0.03) & valid
    true_edges = dilate(object_edges | depth_edges, 1)
    non_edge = valid & ~dilate(true_edges, 4)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 55, 130).astype(bool) & valid
    rgb_texture_edges = rgb_edges & ~dilate(true_edges, 3)

    dfdd_grad = gradient_magnitude(dfdd_depth, valid)
    dfdd_error = np.abs(dfdd_depth - true_depth)

    metrics = {
        "scene_dir": str(args.scene_dir),
        "alignment_dir": str(args.alignment_dir),
        "valid_depth_min_m": args.valid_depth_min,
        "valid_depth_max_m": args.valid_depth_max,
        "used_alignment_valid_mask": valid_mask_path.exists(),
        "dfdd_dense_mae_m": masked_mean(dfdd_error, valid),
        "dfdd_gradient_mean_at_true_depth_edges": masked_mean(dfdd_grad, true_edges),
        "dfdd_gradient_mean_away_from_true_edges": masked_mean(dfdd_grad, non_edge),
        "dfdd_gradient_true_edge_to_non_edge_ratio": masked_mean(dfdd_grad, true_edges)
        / max(masked_mean(dfdd_grad, non_edge), 1e-7),
        "dfdd_gradient_mean_at_rgb_texture_edges_not_depth_edges": masked_mean(dfdd_grad, rgb_texture_edges),
        "dfdd_gradient_texture_to_true_edge_ratio": masked_mean(dfdd_grad, rgb_texture_edges)
        / max(masked_mean(dfdd_grad, true_edges), 1e-7),
        "per_object_depths": per_object_depth_table(object_id, names_by_id, true_depth, dfdd_depth, valid),
        "interpretation": (
            "A real dense depth map should show stronger gradients at true object/depth boundaries than on interior checker texture edges. "
            "Large texture-to-edge ratios or large per-object median errors indicate texture leakage and poor metric generalization."
        ),
    }

    save_figure(
        args.output_dir / "dense_map_edge_check.png",
        rgb,
        true_depth,
        dfdd_depth,
        dfdd_error,
        dfdd_grad,
        true_edges,
        rgb_texture_edges,
        valid,
    )
    (args.output_dir / "dense_map_edge_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
