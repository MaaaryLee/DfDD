"""Render flat checkerboard linear-slide calibration data for SpiderCam/DfDD.

The output naming matches SpiderCam's ``linear_slide_new`` dataset loader.
Unlike ``render_linear_slide.py``, the target texture is a checkerboard.
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "optix", "metal"), default="auto")
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
    candidates = [requested.upper()] if requested != "auto" else ["OPTIX", "CUDA", "METAL"]
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


def checker_material() -> bpy.types.Material:
    material = bpy.data.materials.new("CheckerboardCalibrationTexture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
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
    shader.inputs["Roughness"].default_value = 0.84

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], shader.inputs["Base Color"])
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
    scene.render.film_transparent = False

    look_items = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast", "None"):
        if look in look_items:
            scene.view_settings.look = look
            break

    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = bpy.context.object
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.lens = 35.0
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.05
    camera.data.clip_end = 5.0
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = None
    camera.data.dof.aperture_fstop = ARGS.fstop
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -1.0))
    target = bpy.context.object
    target.name = "CheckerboardSlideTarget"
    target.rotation_euler = (0.0, 0.0, 0.0)
    target.data.materials.append(checker_material())

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.25))
    light = bpy.context.object
    light.data.energy = 22.0
    light.data.shape = "DISK"
    light.data.size = 1.5

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12
    scene.world = world

    if ARGS.engine == "cycles":
        backend = configure_cycles(scene, ARGS.device)
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        backend = "EEVEE"
    scene["compute_backend"] = backend
    return camera, target


def fit_target_to_view(target: bpy.types.Object, depth: float) -> None:
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
        # cam_0/cam_1 feed FocalSplit as I0/I1, and Is = (I0 - I1) / 2 changes sign if
        # they are swapped, which traps the optimiser at a negative depth response.
        render(camera, ARGS.focus_near, ARGS.output_dir / f"cam_0_500_480_{index}.png")
        render(camera, ARGS.focus_far, ARGS.output_dir / f"cam_1_500_480_{index}.png")
        records.append({"index": index, "true_depth_m": depth})

    metadata = {
        "scene": "flat checkerboard linear-slide calibration target",
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
    (ARGS.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


ARGS = parse_args()
main()
