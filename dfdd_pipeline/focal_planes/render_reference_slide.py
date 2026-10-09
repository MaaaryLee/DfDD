"""Render the inputs of the fast focal-plane simulator (run inside Blender).

The calibration slide is a fronto-parallel target that is rescaled at every depth to fill the
view, so the sharp image is the same at every depth and a defocused slide image depends only on
the blur-circle diameter ``b`` (pixels). This script renders, with the scene of
``calibration/render_checkerboard_slide.py``:

1. ``sharp_linear.npy`` / ``sharp.png``: the slide without depth of field, supersampled
   (``--supersample``) and with overscan (``--overscan``) so that blurring near the frame edge sees
   the same target border and background as Blender. The ``.npy`` is scene-linear RGB; the 16-bit
   PNG of the same render carries the view transform (AgX look) the slide renders use, so the
   simulator can fit the tone curve pixel by pixel.
2. ``val_b<b>_s<seed>.png``: depth-of-field renders at the nominal resolution with known blur
   diameters and two sampling seeds, to validate the simulator's mean image and to measure the
   render noise and its correlation between renders that share a seed.

Usage:
    blender --background --python dfdd_pipeline/focal_planes/render_reference_slide.py -- \
        --output-dir artifacts/focal_planes/reference --device optix
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture", choices=("checker", "fractal"), default="fractal")
    parser.add_argument("--depth", type=float, default=3.0, help="Target distance (m); the slide image does not depend on it")
    parser.add_argument("--lens-mm", type=float, default=85.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--fstop", type=float, default=2.0, help="Aperture of the validation renders")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--supersample", type=int, default=4)
    parser.add_argument("--overscan", type=float, default=1.25, help="Rendered field relative to the nominal view")
    parser.add_argument("--sharp-samples", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=2048, help="Samples of the validation renders (the slide renders use 2048)")
    parser.add_argument("--blurs", type=float, nargs="*", default=[0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16],
                        help="Blur-circle diameters (nominal pixels) of the validation renders")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--light-z", type=float, default=-0.25)
    parser.add_argument("--target-margin", type=float, default=1.08)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "optix"), default="auto")
    return parser.parse_args(argv)


ARGS = parse_args()


def configure_cycles(scene: bpy.types.Scene, requested: str) -> str:
    scene.render.engine = "CYCLES"
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    if requested == "cpu":
        scene.cycles.device = "CPU"
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ([requested.upper()] if requested != "auto" else ["OPTIX", "CUDA"]):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            enabled = 0
            for device in prefs.devices:
                device.use = device.type != "CPU"
                enabled += int(device.use)
            if enabled:
                scene.cycles.device = "GPU"
                return backend
        except Exception:
            continue
    scene.cycles.device = "CPU"
    return "CPU"


def slide_material() -> bpy.types.Material:
    """The target material of ``render_checkerboard_slide.py`` (default margin 1.08, so no remapping)."""
    material = bpy.data.materials.new("SlideTexture")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    checker = nodes.new("ShaderNodeTexChecker")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")
    checker.inputs["Color1"].default_value = (0.03, 0.03, 0.03, 1.0)
    checker.inputs["Color2"].default_value = (0.96, 0.96, 0.92, 1.0)
    checker.inputs["Scale"].default_value = 18.0
    texture_scale = ARGS.target_margin / 1.08
    mapping.inputs["Scale"].default_value = (texture_scale, texture_scale, 1.0)
    texture_offset = 0.5 * (1.0 - texture_scale)
    mapping.inputs["Location"].default_value = (texture_offset, texture_offset, 0.0)
    shader.inputs["Roughness"].default_value = 0.84
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    if ARGS.texture == "fractal":
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 9.0
        noise.inputs["Detail"].default_value = 15.0
        noise.inputs["Roughness"].default_value = 0.75
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.33
        ramp.color_ramp.elements[0].color = (0.03, 0.03, 0.03, 1.0)
        ramp.color_ramp.elements[1].position = 0.67
        ramp.color_ramp.elements[1].color = (0.96, 0.96, 0.92, 1.0)
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    else:
        links.new(checker.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    look_items = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast", "None"):
        if look in look_items:
            scene.view_settings.look = look
            break

    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.lens = ARGS.lens_mm
    camera.data.clip_start = 0.05
    camera.data.clip_end = 100.0
    scene.camera = camera

    # Target sized to the NOMINAL view, exactly as fit_target_to_view() does.
    depth = ARGS.depth
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -depth))
    target = bpy.context.object
    vertical_fov = 2.0 * math.atan(ARGS.sensor_width_mm * (ARGS.height / ARGS.width) / (2.0 * ARGS.lens_mm))
    half_height = depth * math.tan(vertical_fov / 2.0) * ARGS.target_margin
    target.scale = (half_height * ARGS.width / ARGS.height, half_height, 1.0)
    target.data.materials.append(slide_material())

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, ARGS.light_z))
    light = bpy.context.object
    light.data.shape = "DISK"
    light.data.size = 1.5
    # --constant-exposure lamp scaling of render_checkerboard_slide.py
    radius, reference = light.data.size / 2.0, 1.0 + ARGS.light_z
    distance = depth + ARGS.light_z
    light.data.energy = 22.0 * (radius ** 2 + distance ** 2) / (radius ** 2 + reference ** 2)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12
    scene.world = world
    backend = configure_cycles(scene, ARGS.device)
    return scene, camera, backend


def set_png16(scene) -> None:
    settings = scene.render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = "RGB"
    settings.color_depth = "16"


def render_sharp(scene, camera) -> dict:
    """No depth of field, supersampled, with overscan; scene-linear .npy and view-transformed PNG."""
    ss, ov = ARGS.supersample, ARGS.overscan
    wide_w, wide_h = round(ARGS.width * ov), round(ARGS.height * ov)
    if (wide_w - ARGS.width) % 2 or (wide_h - ARGS.height) % 2:
        raise SystemExit("--overscan must add an even number of pixels on each axis")
    scene.render.resolution_x, scene.render.resolution_y = wide_w * ss, wide_h * ss
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = ARGS.sensor_width_mm * wide_w / ARGS.width
    camera.data.dof.use_dof = False
    scene.cycles.samples = ARGS.sharp_samples
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.seed = 0
    bpy.ops.render.render()
    result = bpy.data.images["Render Result"]

    exr = ARGS.output_dir / "sharp_linear.exr"
    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    settings.color_mode = "RGB"
    settings.color_depth = "32"
    settings.exr_codec = "ZIP"
    result.save_render(str(exr.resolve()), scene=scene)
    set_png16(scene)
    result.save_render(str((ARGS.output_dir / "sharp.png").resolve()), scene=scene)

    loaded = bpy.data.images.load(str(exr.resolve()))
    loaded.colorspace_settings.name = "Non-Color"  # keep the stored scene-linear values
    pixels = np.array(loaded.pixels[:], dtype=np.float32).reshape(loaded.size[1], loaded.size[0], loaded.channels)
    np.save(ARGS.output_dir / "sharp_linear.npy", np.ascontiguousarray(pixels[::-1, :, :3]))  # top row first, like the PNG
    return {"supersample": ss, "overscan": ov, "wide_size": [wide_w, wide_h],
            "hires_size": [wide_w * ss, wide_h * ss], "samples": ARGS.sharp_samples, "pixel_filter": "BOX 1.0 (hi-res px)"}


def render_validation(scene, camera) -> list[dict]:
    """Nominal camera with depth of field; focus placed so the blur-circle diameter is b pixels."""
    scene.render.resolution_x, scene.render.resolution_y = ARGS.width, ARGS.height
    camera.data.sensor_width = ARGS.sensor_width_mm
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = None
    camera.data.dof.aperture_fstop = ARGS.fstop
    scene.cycles.samples = ARGS.samples
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    set_png16(scene)
    pitch_mm = ARGS.sensor_width_mm / ARGS.width
    k_px_per_diopter = ARGS.lens_mm ** 2 / (ARGS.fstop * pitch_mm) * 1e-3
    records = []
    for blur in ARGS.blurs:
        focus = 1.0 / (1.0 / ARGS.depth + blur / k_px_per_diopter)  # nearer than the target
        camera.data.dof.focus_distance = focus
        for seed in ARGS.seeds:
            scene.cycles.seed = seed
            name = f"val_b{blur:g}_s{seed}.png"
            scene.render.filepath = str((ARGS.output_dir / name).resolve())
            bpy.ops.render.render(write_still=True)
            records.append({"file": name, "blur_px": blur, "seed": seed, "focus_m": focus})
    return records


def main() -> None:
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    scene, camera, backend = build_scene()
    sharp = render_sharp(scene, camera)
    validation = render_validation(scene, camera)
    pitch_mm = ARGS.sensor_width_mm / ARGS.width
    metadata = {
        "texture": ARGS.texture, "depth_m": ARGS.depth, "lens_mm": ARGS.lens_mm,
        "sensor_width_mm": ARGS.sensor_width_mm, "fstop": ARGS.fstop, "resolution": [ARGS.width, ARGS.height],
        "k_px_per_diopter": ARGS.lens_mm ** 2 / (ARGS.fstop * pitch_mm) * 1e-3,
        "target_margin": ARGS.target_margin, "light_z_m": ARGS.light_z,
        "view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
        "validation_samples": ARGS.samples, "compute_backend": backend, "blender_version": bpy.app.version_string,
        "sharp": sharp, "validation": validation,
    }
    (ARGS.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


main()
