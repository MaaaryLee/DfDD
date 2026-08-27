# Generate the brick torus on Windows without Infinigen

This example requires only Blender. Infinigen, Python, `uv`, CUDA, and external
texture files are not required. Blender runs the included Python script with its
own `bpy` Python API.

## 1. Install prerequisites

Install:

1. Git for Windows: <https://git-scm.com/download/win>
2. Blender 4.2 or newer: <https://www.blender.org/download/>

During Git installation, accepting the defaults is sufficient.

## 2. Clone the repository

Open PowerShell and run:

```powershell
cd $HOME\Documents
git clone https://github.com/MaaaryLee/DfDD.git
cd DfDD
git switch codex/rtx3080-calibration-pipeline
```

## 3. Generate the torus

For a standard Blender 4.2 installation:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
& $Blender --background --python .\make_brick_torus_standalone.py -- `
  --output-dir .\outputs\brick_torus `
  --engine cycles `
  --samples 64
```

If a different Blender version is installed, change `$Blender` to the actual
`blender.exe` path. For a quick CPU test, use Eevee:

```powershell
& $Blender --background --python .\make_brick_torus_standalone.py -- `
  --output-dir .\outputs\brick_torus `
  --engine eevee
```

The command creates:

```text
outputs\brick_torus\brick_torus.blend
outputs\brick_torus\brick_torus.png
```

Open the editable scene with:

```powershell
& $Blender .\outputs\brick_torus\brick_torus.blend
```

## Included generation data

The standalone script contains the relevant Infinigen demo parameters directly:

- torus major radius: `0.5 m`;
- torus minor radius: `0.25 m`;
- major/minor segments: `256 / 128`;
- rotation: `(-40°, -25°, 140°)`;
- procedural brick colors, mortar, roughness, and bump;
- checker floor, white scale reference, camera, world, and lighting.

The brick material is implemented with standard Blender shader nodes. It is a
compact approximation of the Infinigen procedural material, not a byte-for-byte
copy of Infinigen's full shader graph.

## 4. Make changes and push from Windows

Do not commit generated `outputs` files. Edit the script or documentation, then:

```powershell
git switch -c your-name/brick-torus-update
git status
git add .\make_brick_torus_standalone.py .\WINDOWS_BRICK_TORUS.md
git commit -m "Update standalone brick torus generator"
git push -u origin your-name/brick-torus-update
```

Then open the GitHub URL printed by `git push` and create a pull request.

If Git asks for identity information:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

GitHub no longer accepts account passwords for command-line pushes. Sign in via
Git Credential Manager when prompted, or use a GitHub personal access token.
