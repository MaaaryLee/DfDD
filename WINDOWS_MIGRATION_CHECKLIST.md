# Windows migration checklist

## Use the prepared archive

Copy `DfDD-Windows-Migration.zip` to the Windows RTX 3080 computer and extract it to a short ASCII-only path such as:

```text
C:\DfDD
```

The archive deliberately excludes Mac virtual environments, Git metadata, scratch files, and generated outputs. Python environments are operating-system-specific and must be recreated on Windows.

## Included and required

- `rtx3080_pipeline/`: full-range rendering, calibration, working-range selection, zoom-in calibration, ablations, and analysis.
- `vendor/SpiderCam/`: the official SpiderCam source plus two required local compatibility fixes. Its nested `.git` metadata is excluded, but its source is included directly.
- `run_rtx3080.ps1`: creates the Windows virtual environment and launches the pipeline.
- `requirements-rtx3080.txt`: non-PyTorch Python dependencies.
- `RTX3080_PIPELINE.md`: setup, commands, outputs, and interpretation notes.
- `defocus_demo.blend`, `make_defocus_demo.py`, `render_defocus_pair.py`, and `DEFOCUS_GUIDE.md`: the original single-pair Blender demonstration. These are useful reference assets; the automated pipeline builds its calibration scene procedurally and does not require external textures.

## Intentionally excluded

- `.venv*`: Mac binaries cannot be reused on Windows.
- `.git` and `vendor/SpiderCam/.git`: history is not needed to run the experiment and nested Git metadata can omit local vendor fixes when republishing.
- `artifacts/`, `outputs/`, `output/`, and `tmp/`: generated or scratch data; the pipeline regenerates its own results.
- `*.blend1`, `.DS_Store`, `__pycache__`, and `*.pyc`: backups and platform-specific caches.

## Windows prerequisites

1. Current NVIDIA driver; verify the RTX 3080 with `nvidia-smi`.
2. Python 3.11 with the `py` launcher.
3. Blender 4.2 or newer.
4. PowerShell.

No full CUDA Toolkit, WSL, Visual Studio, external texture pack, physical SpiderCam driver, `ftd2xx`, or PyQt5 is required for this simulated experiment.

## Run and verify

From PowerShell in the extracted directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_rtx3080.ps1 --engine eevee --full-count 7 --zoom-count 5
```

This fast diagnostic must finish before the default Cycles smoke run:

```powershell
.\run_rtx3080.ps1
```

Successful setup prints an RTX 3080 GPU name and writes presentation-ready figures under:

```text
artifacts\rtx3080_smoke\figures
```

## Integrity check

On Windows, compare the archive hash with the adjacent `.sha256` file:

```powershell
Get-FileHash .\DfDD-Windows-Migration.zip -Algorithm SHA256
```
