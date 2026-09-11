"""Align monocular depth predictions to DfDD sparse anchors and evaluate.

The script runs the trained DfDD calibration on a shaped synthetic scene,
runs Depth Anything V2 on the same RGB image, fits a linear transform
``metric_depth = a * monocular_depth + b`` using high-confidence DfDD pixels,
and compares all maps to Blender ray-cast ground truth.

If a Vision Banana depth map is later exported as a numpy array or grayscale
image, pass it via ``--vision-banana-depth`` and it will go through the same
alignment/evaluation path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPIDERCAM = ROOT / "vendor" / "SpiderCam"
if str(SPIDERCAM) not in sys.path:
    sys.path.insert(0, str(SPIDERCAM))

from models import FocalSplit  # noqa: E402


def select_device() -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU.

    Set DFDD_TORCH_DEVICE to force a backend; MPS still lacks a few operators, so that
    escape hatch matters on Apple silicon.
    """
    override = os.environ.get("DFDD_TORCH_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "artifacts" / "rtx3080_smoke" / "object_scene")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "rtx3080_smoke" / "calibrations" / "zoom_2scale_dxdy" / "checkpoints" / "checkpoint_100epochs.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "rtx3080_smoke" / "monocular_alignment")
    parser.add_argument("--depth-anything-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--vision-banana-depth", type=Path)
    parser.add_argument("--anchor-sparsity", type=float, default=0.9)
    parser.add_argument("--valid-depth-min", type=float)
    parser.add_argument("--valid-depth-max", type=float)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def finite_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask = mask & np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def finite_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask = mask & np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def load_spidercam_gray(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 256.0
    return TF.rgb_to_grayscale(tensor)


def run_dfdd(scene_dir: Path, checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    device = select_device()
    model = FocalSplit(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Must match the linear_slide_new loader used for calibration: img_plus (I0) is the
    # far-focus frame and img_minus (I1) is the near-focus frame. Swapping them flips the
    # sign of Is = (I0 - I1) / 2, and the learned A cannot cross zero to absorb it.
    img_plus = load_spidercam_gray(scene_dir / "cam_1_far_focus.png").unsqueeze(0).to(device)
    img_minus = load_spidercam_gray(scene_dir / "cam_0_near_focus.png").unsqueeze(0).to(device)
    with torch.no_grad():
        z_pred, conf, _z_pred1, _conf1, _outputs = model(img_plus, img_minus)
    return z_pred[0, 0].detach().cpu().numpy(), conf[0, 0].detach().cpu().numpy()


def run_depth_anything(image_path: Path, model_id: str, target_shape: tuple[int, int]) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = select_device()
    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=target_shape,
            mode="bicubic",
            align_corners=False,
        )
    return prediction.squeeze().detach().cpu().numpy().astype(np.float32)


def load_external_depth(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        image = Image.open(path).convert("L").resize((target_shape[1], target_shape[0]), Image.Resampling.BICUBIC)
        depth = np.asarray(image).astype(np.float32)
    if depth.shape != target_shape:
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    return depth.astype(np.float32)


def confidence_mask(conf: np.ndarray, valid: np.ndarray, sparsity: float) -> np.ndarray:
    candidates = valid & np.isfinite(conf)
    if not np.any(candidates):
        return np.zeros_like(valid, dtype=bool)
    threshold = np.quantile(conf[candidates], sparsity)
    return candidates & (conf >= threshold)


def split_mask(mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    n_fit = max(1, int(round(0.7 * len(xs))))
    fit_idx = order[:n_fit]
    eval_idx = order[n_fit:]
    fit = np.zeros_like(mask, dtype=bool)
    held = np.zeros_like(mask, dtype=bool)
    fit[ys[fit_idx], xs[fit_idx]] = True
    if len(eval_idx):
        held[ys[eval_idx], xs[eval_idx]] = True
    else:
        held[ys[fit_idx], xs[fit_idx]] = True
    return fit, held


def linear_align(source: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    mask = fit_mask & np.isfinite(source) & np.isfinite(target)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Need at least two finite anchor pixels for linear alignment")
    x = source[mask].reshape(-1)
    y = target[mask].reshape(-1)
    design = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    return (a * source + b).astype(np.float32), float(a), float(b)


def save_depth_png(path: Path, depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    image = np.ma.masked_where(~valid | ~np.isfinite(depth), depth)
    handle = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    fig.colorbar(handle, ax=ax, shrink=0.85, label="Depth (m)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_error_png(path: Path, pred: np.ndarray, target: np.ndarray, valid: np.ndarray, vmax: float = 0.35) -> None:
    error = np.abs(pred - target)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    image = np.ma.masked_where(~valid | ~np.isfinite(error), error)
    handle = ax.imshow(image, vmin=0.0, vmax=vmax, cmap="magma")
    ax.axis("off")
    fig.colorbar(handle, ax=ax, shrink=0.85, label="Absolute error (m)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overview(output_dir: Path, rgb: np.ndarray, maps: dict[str, np.ndarray], valid: np.ndarray, metrics: dict[str, dict[str, float]]) -> None:
    vmin = float(np.nanpercentile(maps["ground_truth"][valid], 2))
    vmax = float(np.nanpercentile(maps["ground_truth"][valid], 98))
    names = [
        ("input_rgb", "Input RGB"),
        ("ground_truth", "Blender true depth"),
        ("dfdd_dense", "DfDD dense"),
        ("dfdd_sparse_display", "DfDD sparse anchors"),
        ("depth_anything_dfdd_aligned", "Depth Anything aligned to DfDD"),
        ("depth_anything_error", "Depth Anything error"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, names):
        ax.set_title(title)
        ax.axis("off")
        if key == "input_rgb":
            ax.imshow(rgb)
        elif key == "depth_anything_error":
            err = np.abs(maps["depth_anything_dfdd_aligned"] - maps["ground_truth"])
            handle = ax.imshow(np.ma.masked_where(~valid | ~np.isfinite(err), err), cmap="magma", vmin=0.0, vmax=0.35)
            fig.colorbar(handle, ax=ax, shrink=0.78)
        else:
            handle = ax.imshow(np.ma.masked_where(~valid | ~np.isfinite(maps[key]), maps[key]), cmap="viridis", vmin=vmin, vmax=vmax)
            fig.colorbar(handle, ax=ax, shrink=0.78)
    subtitle = (
        f"Depth Anything aligned to DfDD sparse anchors: "
        f"MAE={metrics['depth_anything_dfdd_aligned']['mae_m']:.3f} m; "
        f"DfDD dense MAE={metrics['dfdd_dense']['mae_m']:.3f} m"
    )
    fig.suptitle(subtitle, fontsize=13)
    fig.savefig(output_dir / "alignment_overview.png", dpi=180)
    plt.close(fig)


def normalize01(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32).copy()
    mask = valid & np.isfinite(out)
    lo = np.nanpercentile(out[mask], 2)
    hi = np.nanpercentile(out[mask], 98)
    out = (out - lo) / max(hi - lo, 1e-7)
    return np.clip(out, 0.0, 1.0)


def save_diagnostics(output_dir: Path, maps: dict[str, np.ndarray], valid: np.ndarray, anchor_mask: np.ndarray) -> None:
    vmin = float(np.nanpercentile(maps["ground_truth"][valid], 2))
    vmax = float(np.nanpercentile(maps["ground_truth"][valid], 98))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)

    panels = [
        ("depth_anything_raw_norm", "Depth Anything raw, normalized", "viridis", 0.0, 1.0),
        ("depth_anything_dfdd_aligned", "Aligned to DfDD sparse", "viridis", vmin, vmax),
        ("depth_anything_oracle_sparse_gt_aligned", "Aligned to true depth at same sparse pixels", "viridis", vmin, vmax),
    ]
    for ax, (key, title, cmap, lo, hi) in zip(axes[0], panels):
        ax.set_title(title)
        ax.axis("off")
        handle = ax.imshow(np.ma.masked_where(~valid | ~np.isfinite(maps[key]), maps[key]), cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(handle, ax=ax, shrink=0.78)

    err_dfdd = np.abs(maps["depth_anything_dfdd_aligned"] - maps["ground_truth"])
    err_oracle = np.abs(maps["depth_anything_oracle_sparse_gt_aligned"] - maps["ground_truth"])
    axes[1, 0].set_title("Error after DfDD-anchor alignment")
    axes[1, 0].axis("off")
    handle = axes[1, 0].imshow(np.ma.masked_where(~valid | ~np.isfinite(err_dfdd), err_dfdd), cmap="magma", vmin=0.0, vmax=0.35)
    fig.colorbar(handle, ax=axes[1, 0], shrink=0.78)

    axes[1, 1].set_title("Error after oracle sparse alignment")
    axes[1, 1].axis("off")
    handle = axes[1, 1].imshow(np.ma.masked_where(~valid | ~np.isfinite(err_oracle), err_oracle), cmap="magma", vmin=0.0, vmax=0.35)
    fig.colorbar(handle, ax=axes[1, 1], shrink=0.78)

    ax = axes[1, 2]
    ax.set_title("Raw monocular depth vs true depth")
    sample = valid & np.isfinite(maps["depth_anything_raw"])
    ys, xs = np.nonzero(sample)
    if len(xs) > 6000:
        rng = np.random.default_rng(13)
        chosen = rng.choice(len(xs), size=6000, replace=False)
        ys, xs = ys[chosen], xs[chosen]
    ax.scatter(maps["depth_anything_raw"][ys, xs], maps["ground_truth"][ys, xs], s=2, alpha=0.16, label="all valid pixels")
    ay, axx = np.nonzero(anchor_mask & np.isfinite(maps["depth_anything_raw"]))
    ax.scatter(maps["depth_anything_raw"][ay, axx], maps["ground_truth"][ay, axx], s=4, alpha=0.35, label="DfDD sparse anchors")
    ax.set_xlabel("Depth Anything raw output")
    ax.set_ylabel("Blender true depth (m)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_dir / "alignment_diagnostics.png", dpi=180)
    plt.close(fig)


def evaluate_monocular(
    name: str,
    raw_depth: np.ndarray,
    reference_depth: np.ndarray,
    true_depth: np.ndarray,
    valid: np.ndarray,
    fit_mask: np.ndarray,
    heldout_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    aligned, a, b = linear_align(raw_depth, reference_depth, fit_mask)
    metrics = {
        "linear_a": a,
        "linear_b": b,
        "mae_m": finite_mae(aligned, true_depth, valid),
        "rmse_m": finite_rmse(aligned, true_depth, valid),
        "anchor_fit_mae_to_reference_m": finite_mae(aligned, reference_depth, fit_mask),
        "anchor_heldout_mae_to_reference_m": finite_mae(aligned, reference_depth, heldout_mask),
        "valid_pixels": int(np.count_nonzero(valid)),
    }
    return aligned, metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    true_depth = np.load(args.scene_dir / "depth_true_m.npy")
    valid = np.isfinite(true_depth)
    h, w = true_depth.shape
    interior = np.zeros_like(valid, dtype=bool)
    interior[11:-11, 11:-11] = True
    valid = valid & interior
    if args.valid_depth_min is not None:
        valid &= true_depth >= args.valid_depth_min
    if args.valid_depth_max is not None:
        valid &= true_depth <= args.valid_depth_max

    rgb_image = np.asarray(Image.open(args.scene_dir / "rgb_reference.png").convert("RGB"))
    dfdd_depth, dfdd_conf = run_dfdd(args.scene_dir, args.checkpoint)
    dfdd_valid = valid & np.isfinite(dfdd_depth) & np.isfinite(dfdd_conf)
    anchor_mask = confidence_mask(dfdd_conf, dfdd_valid, args.anchor_sparsity)
    fit_mask, heldout_mask = split_mask(anchor_mask, args.seed)

    da_raw = run_depth_anything(args.scene_dir / "rgb_reference.png", args.depth_anything_model, (h, w))
    da_dfdd_aligned, da_dfdd_metrics = evaluate_monocular(
        "depth_anything", da_raw, dfdd_depth, true_depth, valid, fit_mask, heldout_mask
    )
    da_oracle_aligned, da_oracle_metrics = evaluate_monocular(
        "depth_anything_oracle", da_raw, true_depth, true_depth, valid, fit_mask, heldout_mask
    )

    metrics = {
        "experiment": "Depth Anything / Vision Banana-style linear alignment to DfDD sparse anchors",
        "scene_dir": str(args.scene_dir),
        "checkpoint": str(args.checkpoint),
        "valid_depth_min_m": args.valid_depth_min,
        "valid_depth_max_m": args.valid_depth_max,
        "anchor_sparsity": args.anchor_sparsity,
        "anchor_pixels_total": int(np.count_nonzero(anchor_mask)),
        "anchor_pixels_fit": int(np.count_nonzero(fit_mask)),
        "anchor_pixels_heldout": int(np.count_nonzero(heldout_mask)),
        "dfdd_dense": {
            "mae_m": finite_mae(dfdd_depth, true_depth, valid),
            "rmse_m": finite_rmse(dfdd_depth, true_depth, valid),
            "valid_pixels": int(np.count_nonzero(dfdd_valid)),
        },
        "dfdd_sparse_anchors": {
            "mae_m": finite_mae(dfdd_depth, true_depth, anchor_mask),
            "rmse_m": finite_rmse(dfdd_depth, true_depth, anchor_mask),
            "valid_pixels": int(np.count_nonzero(anchor_mask)),
        },
        "depth_anything_dfdd_aligned": da_dfdd_metrics,
        "depth_anything_oracle_sparse_gt_aligned": {
            **da_oracle_metrics,
            "note": "Diagnostic only: fits the same sparse pixel locations to Blender truth instead of DfDD, showing the best-case value of a reliable sparse metric anchor.",
        },
        "vision_banana": {
            "status": "not_run_no_local_checkpoint_or_api_output",
            "note": "Pass --vision-banana-depth with an exported Vision Banana metric/depth image or .npy to run the same alignment/evaluation.",
        },
    }

    maps = {
        "ground_truth": true_depth,
        "dfdd_dense": dfdd_depth,
        "dfdd_sparse_display": np.where(anchor_mask, dfdd_depth, np.nan),
        "depth_anything_raw": da_raw,
        "depth_anything_raw_norm": normalize01(da_raw, valid),
        "depth_anything_dfdd_aligned": da_dfdd_aligned,
        "depth_anything_oracle_sparse_gt_aligned": da_oracle_aligned,
    }

    if args.vision_banana_depth:
        vb_raw = load_external_depth(args.vision_banana_depth, (h, w))
        vb_aligned, vb_metrics = evaluate_monocular(
            "vision_banana", vb_raw, dfdd_depth, true_depth, valid, fit_mask, heldout_mask
        )
        maps["vision_banana_raw"] = vb_raw
        maps["vision_banana_aligned"] = vb_aligned
        metrics["vision_banana"] = vb_metrics

    np.save(args.output_dir / "dfdd_depth_m.npy", dfdd_depth)
    np.save(args.output_dir / "dfdd_conf.npy", dfdd_conf)
    np.save(args.output_dir / "dfdd_sparse_anchor_mask.npy", anchor_mask)
    np.save(args.output_dir / "valid_eval_mask.npy", valid)
    np.save(args.output_dir / "depth_anything_raw.npy", da_raw)
    np.save(args.output_dir / "depth_anything_dfdd_aligned_m.npy", da_dfdd_aligned)
    np.save(args.output_dir / "depth_anything_oracle_sparse_gt_aligned_m.npy", da_oracle_aligned)
    np.save(args.output_dir / "depth_anything_aligned_m.npy", da_dfdd_aligned)
    if "vision_banana_aligned" in maps:
        np.save(args.output_dir / "vision_banana_aligned_m.npy", maps["vision_banana_aligned"])

    vmin = float(np.nanpercentile(true_depth[valid], 2))
    vmax = float(np.nanpercentile(true_depth[valid], 98))
    save_depth_png(args.output_dir / "ground_truth_depth.png", true_depth, valid, vmin, vmax)
    save_depth_png(args.output_dir / "dfdd_dense_depth.png", dfdd_depth, valid, vmin, vmax)
    save_depth_png(args.output_dir / "dfdd_sparse_anchors.png", np.where(anchor_mask, dfdd_depth, np.nan), anchor_mask, vmin, vmax)
    save_depth_png(args.output_dir / "depth_anything_aligned_depth.png", da_dfdd_aligned, valid, vmin, vmax)
    save_depth_png(args.output_dir / "depth_anything_oracle_sparse_gt_aligned_depth.png", da_oracle_aligned, valid, vmin, vmax)
    save_error_png(args.output_dir / "dfdd_dense_error.png", dfdd_depth, true_depth, valid)
    save_error_png(args.output_dir / "depth_anything_aligned_error.png", da_dfdd_aligned, true_depth, valid)
    save_error_png(args.output_dir / "depth_anything_oracle_sparse_gt_aligned_error.png", da_oracle_aligned, true_depth, valid)
    save_overview(args.output_dir, rgb_image, maps, valid, metrics)
    save_diagnostics(args.output_dir, maps, valid, anchor_mask)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
