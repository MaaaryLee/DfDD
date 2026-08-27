"""Render a differential-defocus pair from the currently opened Blender scene.

Usage:
  Blender --background scene.blend --python render_defocus_pair.py -- \
    --focus-a 4.8 --focus-b 5.2 --fstop 1.4 --output-dir outputs/pair

Distances are Blender scene units. With unit scale 1.0, treat them as meters.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-a", type=float, required=True)
    parser.add_argument("--focus-b", type=float, required=True)
    parser.add_argument("--fstop", type=float, default=1.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    return parser.parse_args(argv)


def render_at_focus(camera, focus_distance, path):
    dof = camera.data.dof
    dof.use_dof = True
    dof.focus_object = None  # An assigned object overrides focus_distance.
    dof.focus_distance = focus_distance

    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError("The scene must have an active camera")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # Cycles gives physically based thin-lens depth of field.
    scene.render.engine = "BLENDER_EEVEE_NEXT" if args.samples == 0 else "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False

    camera.data.dof.aperture_fstop = args.fstop
    render_at_focus(camera, args.focus_a, args.output_dir / "defocus_a.png")
    render_at_focus(camera, args.focus_b, args.output_dir / "defocus_b.png")

    metadata = {
        "focus_distance_a_scene_units": args.focus_a,
        "focus_distance_b_scene_units": args.focus_b,
        "aperture_fstop": args.fstop,
        "focal_length_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
        "sensor_height_mm": camera.data.sensor_height,
        "camera": camera.name,
        "blender_version": bpy.app.version_string,
    }
    (args.output_dir / "camera.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


main()
