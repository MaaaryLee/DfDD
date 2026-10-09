"""Do slide-optimal focal planes also give better DfDD depth in the indoor rooms?

For each focal pair: calibrate FocalSplit on that pair's Blender slide sweep (converged trainer),
save the checkpoint in the trainer's format, run it on the room views rendered with the same pair,
and score DfDD depth against the Cycles depth pass. Only DfDD is scored (no monocular model).

Pixels scored: valid depth inside the calibrated range [z_lo, z_hi]. Reported per view and pooled
over views: dense MAE, MAE of the top-10 % confidence pixels (90 % sparse, quantile taken over the
in-range pixels of the view), and both relative to depth.

Usage:
    python dfdd_pipeline/focal_planes/room_check.py \
        --pair current 1.984 4.416 artifacts/rtx3080_85mm_f2/data/zoom_range artifacts/indoor_85mm_f2 \
        --pair inward 2.3 4.0 artifacts/focal_planes/blender/f2_2p08to4p32/focus_2.300_4.000/zoom_range artifacts/focal_planes/rooms/focus_2.300_4.000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dfdd_pipeline.common.paths import ROOT  # noqa: E402
from dfdd_pipeline.focal_planes.check_calibration import load_blender_sweep  # noqa: E402
from dfdd_pipeline.focal_planes.simulator import DEVICE, LOADER_GRAY, calibrate, focal_split_config  # noqa: E402
from dfdd_pipeline.monocular.pair_denoising import denoise_pair  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    """The slide loader's input: grey levels (16-bit PNGs at full precision) / 256."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    bgr = raw.astype(np.float32) / (257.0 if raw.dtype == np.uint16 else 1.0)
    gray = LOADER_GRAY[0] * bgr[..., 0] + LOADER_GRAY[1] * bgr[..., 1] + LOADER_GRAY[2] * bgr[..., 2]
    return gray / 256.0


def views_under(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/*/cam_0_near_focus.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=5, action="append", required=True, metavar=("LABEL", "NEAR", "FAR", "SLIDES", "VIEWS"))
    parser.add_argument("--z-lo", type=float, default=2.08)
    parser.add_argument("--z-hi", type=float, default=4.32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--pair-denoise", action="store_true",
                        help="Rebuild each focus pair with monocular/pair_denoising.py before running the model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/focal_planes/rooms/room_check.json")
    args = parser.parse_args()
    config = focal_split_config()
    report = {"z_lo": args.z_lo, "z_hi": args.z_hi, "pair_denoise": args.pair_denoise, "pairs": []}
    view_names = None
    for label, near, far, slides, views_root in args.pair:
        near, far = float(near), float(far)
        ip, im, z, meta = load_blender_sweep(Path(slides))
        result = calibrate(ip, im, z, config, steps=args.steps, init="physics", b_value=0.5 * (1 / near + 1 / far), schedule="cosine")
        del ip, im
        model = result.model.eval()
        ckpt_dir = args.output.parent / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": OmegaConf.to_container(config, resolve=True), "epoch": args.steps,
                    "model_state_dict": model.state_dict(), "focus_m": [near, far], "slides": str(slides)},
                   ckpt_dir / f"{label}_{near:.3f}_{far:.3f}.pth")
        views = views_under(Path(views_root))
        names = [f"{v.parent.name}/{v.name}" for v in views]
        if view_names is None:
            view_names = names
        rows, pooled = [], {"dense": [], "sparse": [], "dense_rel": [], "sparse_rel": []}
        for view, name in zip(views, names):
            if name not in view_names:
                continue
            far_img, near_img = load_gray(view / "cam_1_far_focus.png"), load_gray(view / "cam_0_near_focus.png")
            if args.pair_denoise:
                far_img, near_img = denoise_pair(far_img, near_img)
            i_plus = torch.from_numpy(np.ascontiguousarray(far_img, dtype=np.float32))[None, None].to(DEVICE)
            i_minus = torch.from_numpy(np.ascontiguousarray(near_img, dtype=np.float32))[None, None].to(DEVICE)
            with torch.no_grad():
                z_pred, conf, *_ = model(i_plus, i_minus)
            z_pred, conf = z_pred[0, 0].cpu().numpy(), conf[0, 0].cpu().numpy()
            truth = np.load(view / "depth_true_m.npy")
            ok = np.isfinite(truth) & (truth >= args.z_lo) & (truth <= args.z_hi) & np.isfinite(z_pred)
            ok[:11, :] = ok[-11:, :] = False
            ok[:, :11] = ok[:, -11:] = False
            if ok.sum() < 100:
                continue
            err = np.abs(z_pred - truth)
            keep = ok & (conf >= np.quantile(conf[ok], 0.9))
            row = {"view": name, "in_range_pixels": int(ok.sum()), "dense_mae_m": float(err[ok].mean()),
                   "sparse_mae_m": float(err[keep].mean()), "dense_rel": float((err / truth)[ok].mean()),
                   "sparse_rel": float((err / truth)[keep].mean())}
            rows.append(row)
            pooled["dense"].append(err[ok]); pooled["sparse"].append(err[keep])
            pooled["dense_rel"].append((err / truth)[ok]); pooled["sparse_rel"].append((err / truth)[keep])
        summary = {k: float(np.concatenate(v).mean()) for k, v in pooled.items()}
        summary.update({f"view_mean_{k}": float(np.mean([r[k] for r in rows])) for k in ("dense_mae_m", "sparse_mae_m", "dense_rel", "sparse_rel")})
        report["pairs"].append({"label": label, "near_m": near, "far_m": far, "views": rows, "pooled": summary,
                                "slide_final_loss": result.losses[-1]})
        print(f"{label} {near}/{far}: {len(rows)} views; pooled in-range MAE dense {100 * summary['dense']:.1f} cm "
              f"({100 * summary['dense_rel']:.1f}%), 90% sparse {100 * summary['sparse']:.1f} cm ({100 * summary['sparse_rel']:.1f}%)", flush=True)
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
