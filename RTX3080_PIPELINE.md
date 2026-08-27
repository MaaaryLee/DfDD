# SpiderCam RTX 3080 calibration smoke pipeline

This repository contains a reproducible, presentation-oriented smoke experiment:

1. render a wide simulated linear-slide depth sweep;
2. calibrate the paper-style two-scale + dx/dy DfDD model;
3. find the longest sampled interval below the paper's 10% relative-error criterion;
4. render a denser sweep around that interval;
5. independently recalibrate four `n_scales` / `dxdy` ablations;
6. export heatmaps, error-vs-true-depth curves, sparsity comparisons, JSON metrics, and an ablation CSV/bar chart.

The default is deliberately small (320×240, 11 full-range depths, 9 zoom depths, 32 Cycles samples). It validates the entire requested process without pretending to be the final paper-scale experiment.

## Windows + RTX 3080 quick start

Requirements:

- NVIDIA RTX 3080 with a current NVIDIA driver;
- Python 3.11 (recommended);
- Blender 4.2 or newer (the included example scene was verified with 4.2);
- Git.

In PowerShell, from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_rtx3080.ps1
```

The script creates `.venv`, installs CUDA PyTorch plus the lightweight analysis dependencies, verifies CUDA, and runs the experiment. If Blender is installed in a nonstandard location:

```powershell
.\run_rtx3080.ps1 --blender "D:\Apps\Blender\blender.exe"
```

To force CUDA instead of OptiX for Blender:

```powershell
.\run_rtx3080.ps1 --render-device cuda
```

## Faster diagnostic run

Use Eevee and fewer images to check installation and file flow:

```powershell
.\run_rtx3080.ps1 --engine eevee --full-count 7 --zoom-count 5
```

This still performs full-range calibration, working-region selection, zoom-in calibration, and all four ablations. It is a pipeline test, not a quantitative result suitable for comparing against the paper.

## Higher-quality follow-up

After the smoke result is verified:

```powershell
.\run_rtx3080.ps1 --samples 128 --width 480 --height 400 --full-count 31 --zoom-count 31 --n-rings 16
```

This is slower and uses the paper's 16 radial zones. Increase one dimension at a time so errors can be traced to a specific change.

## Outputs

Generated files are intentionally git-ignored and written to:

```text
artifacts/rtx3080_smoke/
├── data/
│   ├── full_range/
│   └── zoom_range/
├── calibrations/
├── figures/
│   ├── full_2scale_dxdy/
│   ├── zoom_1scale_no_dxdy/
│   ├── zoom_1scale_dxdy/
│   ├── zoom_2scale_no_dxdy/
│   ├── zoom_2scale_dxdy/
│   ├── ablation_comparison.png
│   └── ablation_summary.csv
└── experiment_manifest.json
```

Each configuration folder contains:

- `error_vs_true_depth.png`: MAE at several sparsities with the 10% working-range threshold;
- `error_vs_sparsity.png`: accuracy-density trade-off;
- `heatmap_sparsity_00.png` and `heatmap_sparsity_90.png`: predicted vs. true depth;
- `summary.json`: numerical values used in the figures.

## Interpretation and caveats

- Every ablation is calibrated independently. Reusing two-scale parameters for a one-scale model would be an invalid comparison.
- `90% sparsity` means retaining the highest-confidence 10% of pixels per depth image.
- If no sampled interval meets the formal 10% criterion during the smoke run, the pipeline zooms around the three lowest-error depths and records the region as a fallback. This prevents a failed early calibration from silently being reported as a valid working range.
- Blender ground-truth depth is in metres. The generated target remains fronto-parallel and fills the view at every slide position.
- The procedural texture and synthetic thin-lens model are for controlled validation. They do not reproduce all aberrations and sensor noise of the physical SpiderCam.

## Important vendor fixes included

Two small compatibility corrections are applied to the vendored SpiderCam code:

- spatial optical-parameter objects use `nn.ModuleList` (they are modules, not raw parameters);
- the calibration DataLoader uses zero worker subprocesses, which is reliable on Windows and appropriate for this small full-batch dataset.
