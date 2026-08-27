"""Blender-side renderer for a simulated differential-defocus linear slide.

Run via Blender, not regular Python. The output names match SpiderCam's
``linear_slide_new`` dataset loader.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths", type=float, nargs="+", required=True)
    parser.add_argument("--focus-near", type=float, default=0.55)
    parser.add_argument("--focus-far", type=float, default=0.95)
    parser.add_argument("--fstop", type=float, default=1.4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "optix"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def configure_cycles(scene: bpy.types.Scene, requested: str) -> str:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = ARGS.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    if requested == "cpu":
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    candidates = [requested.upper()] if requested != "auto" else ["OPTIX", "CUDA"]
    for backend in candidates:
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


def material_with_texture() -> bpy.types.Material:
    material = bpy.data.materials.new("CalibrationTexture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    mix = nodes.new("ShaderNodeMixRGB")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")

    noise.inputs["Scale"].default_value = 38.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.72
    voronoi.inputs["Scale"].default_value = 21.0
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 0.68
    shader.inputs["Roughness"].default_value = 0.82

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    links.new(noise.outputs["Fac"], mix.inputs[1])
    links.new(voronoi.outputs["Distance"], mix.inputs[2])
    links.new(mix.outputs[0], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = ARGS.width
    scene.render.resolution_y = ARGS.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    # Blender 4.2 prefixes AgX look names; newer builds may expose the short
    # form. Select a supported value instead of tying the script to one build.
    look_items = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast", "None"):
        if look in look_items:
            scene.view_settings.look = look
            break
    scene.render.film_transparent = False

    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.lens = 35.0
    camera.data.sensor_width = 36.0
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = None
    camera.data.dof.aperture_fstop = ARGS.fstop
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -1.0))
    target = bpy.context.object
    target.name = "LinearSlideTarget"
    target.rotation_euler = (0.0, 0.0, 0.0)
    target.data.materials.append(material_with_texture())

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.25))
    light = bpy.context.object
    light.data.energy = 20.0
    light.data.shape = "DISK"
    light.data.size = 1.5
    # Area lights emit along local -Z by default, toward the target plane.

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.18
    scene.world = world

    backend = "EEVEE"
    if ARGS.engine == "cycles":
        backend = configure_cycles(scene, ARGS.device)
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene["compute_backend"] = backend
    return camera, target


def fit_target_to_view(target: bpy.types.Object, depth: float) -> None:
    # Keep a textured fronto-parallel plane filling the view at every depth.
    target.location = Vector((0.0, 0.0, -depth))
    vertical_fov = 2.0 * math.atan(36.0 * (ARGS.height / ARGS.width) / (2.0 * 35.0))
    half_height = depth * math.tan(vertical_fov / 2.0) * 1.08
    target.scale = (half_height * ARGS.width / ARGS.height, half_height, 1.0)


def render(camera: bpy.types.Object, focus: float, path: Path) -> None:
    camera.data.dof.focus_distance = focus
    bpy.context.scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)


def main() -> None:
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    camera, target = build_scene()
    records = []
    for index, depth in enumerate(ARGS.depths):
        fit_target_to_view(target, depth)
        # camera0 is near-focus (+), camera1 is far-focus (-), matching loader order.
        render(camera, ARGS.focus_near, ARGS.output_dir / f"cam_1_500_480_{index}.png")
        render(camera, ARGS.focus_far, ARGS.output_dir / f"cam_0_500_480_{index}.png")
        records.append({"index": index, "true_depth_m": depth})

    metadata = {
        "depths_m": list(ARGS.depths),
        "focus_near_m": ARGS.focus_near,
        "focus_far_m": ARGS.focus_far,
        "fstop": ARGS.fstop,
        "resolution": [ARGS.width, ARGS.height],
        "samples": ARGS.samples,
        "engine": ARGS.engine,
        "compute_backend": bpy.context.scene["compute_backend"],
        "blender_version": bpy.app.version_string,
        "records": records,
    }
    (ARGS.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


ARGS = parse_args()
main()
