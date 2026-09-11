"""End-to-end RTX 3080 SpiderCam smoke experiment orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from analyze import analyze_one, comparison_plot, longest_working_interval, per_depth_metrics
from platform_support import find_blender


ROOT = Path(__file__).resolve().parents[1]
SPIDERCAM = ROOT / "vendor" / "SpiderCam"
ARTIFACTS = ROOT / "artifacts" / "rtx3080_smoke"


def depth_grid(start: float, stop: float, count: int) -> list[float]:
    return [round(float(value), 6) for value in np.linspace(start, stop, count)]


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def render_dataset(args, name: str, depths: list[float]) -> Path:
    output = ARTIFACTS / "data" / name
    command = [
        find_blender(args.blender), "--background", "--python",
        str(ROOT / "rtx3080_pipeline" / "render_linear_slide.py"), "--",
        "--output-dir", str(output), "--depths", *map(str, depths),
        "--focus-near", str(args.focus_near), "--focus-far", str(args.focus_far),
        "--fstop", str(args.fstop), "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.samples), "--engine", args.engine, "--device", args.render_device,
    ]
    run(command)
    return output


def write_data_config(name: str, data_dir: Path, depths: list[float], args) -> str:
    if len(depths) < 2:
        raise ValueError("At least two depths are required")
    step = depths[1] - depths[0]
    config_name = f"rtx_{name}"
    config = {
        "name": config_name,
        "data_path": str(data_dir.parent.resolve()),
        "type": "linear_slide_new",
        "group": data_dir.name,
        "channel": "gray",
        "crop": True,
        "frame_range": [0, len(depths)],
        "mask": [0, args.height, 0, args.width],
        "center": [args.height // 2, args.width // 2],
        "step_size": float(step),
        "offset": float(depths[0]),
    }
    path = SPIDERCAM / "configs" / "data" / f"{config_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_name


def spidercam_result_dir(config_name: str, dataset_group: str, scales: int, dxdy: bool, args) -> Path:
    model = f"{scales}scales" + ("_dxdy" if dxdy else "") + f"_joint_{args.n_rings}rings_conf=VW_5x5gaussian"
    optim = f"adamw_lr={args.learning_rate}_wd=0.005_clip=1.0_sparsity={args.calibration_sparsity}"
    return SPIDERCAM / "results" / f"{config_name}_{dataset_group}_gray_cropped" / f"{model}_{optim}"


def calibrate(config_name: str, dataset_group: str, label: str, scales: int, dxdy: bool, args) -> Path:
    command = [
        sys.executable, "train_focal_split.py", f"data={config_name}",
        f"model.n_scales={scales}", f"model.dxdy={str(dxdy).lower()}",
        "model.mode=joint", "model.const=rings", f"model.n_rings={args.n_rings}",
        "model.conf=VW", f"optim.lr={args.learning_rate}",
        f"optim.batch_size={args.batch_size}", f"optim.sparsity={args.calibration_sparsity}",
        f"n_epochs={args.epochs}", f"eval_interval={max(1, args.epochs // 5)}",
    ]
    run(command, SPIDERCAM)
    result = spidercam_result_dir(config_name, dataset_group, scales, dxdy, args)
    if not (result / "z_true.npy").exists():
        raise FileNotFoundError(f"Expected calibration outputs were not created: {result}")
    destination = ARTIFACTS / "calibrations" / label
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(result, destination)
    return destination


def working_range(result_dir: Path, sparsity: float) -> tuple[float, float]:
    z_true = np.load(result_dir / "z_true.npy")[..., 11:-11, 11:-11]
    z_pred = np.load(result_dir / "z_pred.npy")[..., 11:-11, 11:-11]
    conf = np.load(result_dir / "conf.npy")[..., 11:-11, 11:-11]
    depth, mae, _ = per_depth_metrics(z_true, z_pred, conf, sparsity)
    interval = longest_working_interval(depth, mae)
    if interval is not None and interval[1] > interval[0]:
        return interval
    # A smoke experiment may not cross the formal 10% threshold. Continue
    # around the three best sampled depths, while recording this fallback.
    best = np.argsort(mae)[: min(3, len(depth))]
    return float(depth[best].min()), float(depth[best].max())


def write_manifest(args, full_depths, zoom_depths, full_wr, summaries) -> None:
    manifest = {
        "experiment": "SpiderCam simulated linear-slide smoke calibration",
        "device_expected": "NVIDIA RTX 3080",
        "full_depths_m": full_depths,
        "full_working_range_or_best_region_m": full_wr,
        "zoom_depths_m": zoom_depths,
        "settings": vars(args),
        "ablation_summaries": summaries,
        "interpretation_note": "Smoke-scale results validate the pipeline; increase depth counts and render samples for final quantitative claims.",
    }
    serializable = {k: str(v) if isinstance(v, Path) else v for k, v in manifest.items()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "experiment_manifest.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender")
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--render-device", choices=("auto", "cpu", "cuda", "optix", "metal"), default="auto")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--focus-near", type=float, default=0.55)
    parser.add_argument("--focus-far", type=float, default=0.95)
    parser.add_argument("--fstop", type=float, default=1.4)
    parser.add_argument("--full-min", type=float, default=0.30)
    parser.add_argument("--full-max", type=float, default=1.30)
    parser.add_argument("--full-count", type=int, default=11)
    parser.add_argument("--zoom-count", type=int, default=9)
    parser.add_argument("--zoom-margin", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100, help="Keep 100 unless SpiderCam checkpoint saving is changed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--calibration-sparsity", type=float, default=0.9)
    parser.add_argument("--n-rings", type=int, default=4)
    args = parser.parse_args()
    if args.epochs % 100:
        parser.error("--epochs must be a multiple of 100 because the upstream trainer saves arrays every 100 epochs")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    full_depths = depth_grid(args.full_min, args.full_max, args.full_count)
    full_data = render_dataset(args, "full_range", full_depths)
    full_config = write_data_config("full_range", full_data, full_depths, args)
    full_result = calibrate(full_config, full_data.name, "full_2scale_dxdy", 2, True, args)
    full_wr = working_range(full_result, args.calibration_sparsity)

    zoom_min = max(args.full_min, full_wr[0] - args.zoom_margin)
    zoom_max = min(args.full_max, full_wr[1] + args.zoom_margin)
    if zoom_max - zoom_min < 0.2:
        center = (zoom_min + zoom_max) / 2.0
        zoom_min = max(args.full_min, center - 0.15)
        zoom_max = min(args.full_max, center + 0.15)
    zoom_depths = depth_grid(zoom_min, zoom_max, args.zoom_count)
    zoom_data = render_dataset(args, "zoom_range", zoom_depths)
    zoom_config = write_data_config("zoom_range", zoom_data, zoom_depths, args)

    result_specs = [("full_2scale_dxdy", full_result)]
    for scales, dxdy in ((1, False), (1, True), (2, False), (2, True)):
        label = f"zoom_{scales}scale_{'dxdy' if dxdy else 'no_dxdy'}"
        result_specs.append((label, calibrate(zoom_config, zoom_data.name, label, scales, dxdy, args)))

    figure_root = ARTIFACTS / "figures"
    summaries = [analyze_one(path, figure_root / label, label) for label, path in result_specs]
    comparison_plot(summaries[1:], figure_root)
    write_manifest(args, full_depths, zoom_depths, full_wr, summaries)
    print(f"\nComplete. Presentation-ready outputs: {figure_root}")


if __name__ == "__main__":
    main()
