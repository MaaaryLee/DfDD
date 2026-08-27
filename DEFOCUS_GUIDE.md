# Defocus pairs with Infinigen and Blender 4.2

## What the pair represents

Keep scene geometry, camera pose, focal length, sensor size, aperture, lighting,
exposure, and random seed fixed. Change only the focus setting:

- `defocus_a.png`: focus distance `s_a`
- `defocus_b.png`: focus distance `s_b = s_a + delta`
- ground-truth depth: render separately without photographic DOF

The core Blender API is:

```python
cam = bpy.context.scene.camera
cam.data.dof.use_dof = True
cam.data.dof.focus_object = None
cam.data.dof.focus_distance = 5.0
cam.data.dof.aperture_fstop = 1.8
```

`focus_object` must be `None`; otherwise Blender uses the object's distance and
ignores the numeric `focus_distance`.

## Render a pair from an Infinigen scene

After Infinigen has produced a populated `scene.blend`, run:

```bash
/Users/maaary/Documents/ChatGPT/DfDD/tmp/infinigen/Blender.app/Contents/MacOS/Blender \
  --background /absolute/path/to/scene.blend \
  --python /Users/maaary/Documents/ChatGPT/DfDD/render_defocus_pair.py -- \
  --focus-a 4.8 --focus-b 5.2 --fstop 1.8 \
  --output-dir /absolute/path/to/pair
```

This writes `defocus_a.png`, `defocus_b.png`, and `camera.json`.

Infinigen's standard nature config sets `render_image.use_dof = False`. If using
Infinigen's own render task instead of this script, override it with:

```text
-p render_image.use_dof=True render_image.dof_aperture_fstop=1.8
```

## Sensor-to-lens distance

Blender 4.2 does **not** expose physical sensor-to-lens separation. These are
different parameters:

- `cam.data.lens`: focal length in millimeters
- `cam.data.sensor_width` / `sensor_height`: sensor dimensions in millimeters
- `cam.data.dof.focus_distance`: object-side focus distance in scene units
- `cam.data.dof.aperture_fstop`: aperture ratio

For an ideal thin lens, object distance `s`, focal length `f`, and image distance
`v` (lens principal plane to sensor) satisfy:

```text
1/f = 1/s + 1/v
v = f*s/(s-f)
s = f*v/(v-f)
```

Use one unit consistently in these equations, normally millimeters. For example,
with `f = 50 mm` and focus at `s = 5000 mm`, the implied image distance is about
`v = 50.505 mm`.

If an experiment specifies two sensor distances `v_a` and `v_b`, convert them to
Blender focus distances:

```python
def sensor_distance_to_focus_distance(v_mm, f_mm):
    if v_mm <= f_mm:
        raise ValueError("A real positive focus requires v > f")
    s_mm = f_mm * v_mm / (v_mm - f_mm)
    return s_mm / 1000.0  # scene units, assuming 1 unit = 1 meter
```

This reproduces the corresponding **focus-plane change** under an ideal thin-lens
assumption. It does not literally translate Blender's virtual sensor, and therefore
does not reproduce every projection/magnification effect of a physical sensor shift.
For exact sensor-translation optics, use a custom camera model or explicit lens
geometry and validate it against the DfDD paper's forward model.
