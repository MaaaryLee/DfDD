"""Render a multi-depth checkerboard scene for DfDD/monocular alignment tests."""

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
    parser.add_argument("--focus-near", type=float, default=0.55)
    parser.add_argument("--focus-far", type=float, default=0.95)
    parser.add_argument("--fstop", type=float, default=1.4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=64)
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


def checker_material(name: str, color_a: tuple[float, float, float, float], color_b: tuple[float, float, float, float], scale: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
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

    checker.inputs["Color1"].default_value = color_a
    checker.inputs["Color2"].default_value = color_b
    checker.inputs["Scale"].default_value = scale
    shader.inputs["Roughness"].default_value = 0.84

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_camera(scene: bpy.types.Scene) -> bpy.types.Object:
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
    return camera


def add_card(
    name: str,
    size: float,
    location: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    material: bpy.types.Material,
    x_scale: float = 1.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = tuple(math.radians(value) for value in rotation_deg)
    obj.scale.x = x_scale
    obj.data.materials.append(material)
    return obj


def build_scene() -> bpy.types.Object:
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

    camera = add_camera(scene)

    black = (0.02, 0.02, 0.02, 1.0)
    white = (0.95, 0.95, 0.90, 1.0)
    blue = (0.18, 0.42, 0.92, 1.0)
    green = (0.15, 0.74, 0.36, 1.0)
    rose = (0.96, 0.30, 0.34, 1.0)

    background_mat = checker_material("background_checker_1p25m", white, (0.70, 0.70, 0.62, 1.0), 18.0)
    near_mat = checker_material("near_checker_0p75m", black, white, 14.0)
    mid_mat = checker_material("mid_checker_0p95m", blue, white, 16.0)
    far_mat = checker_material("far_checker_1p08m", green, black, 18.0)
    edge_mat = checker_material("edge_checker_1p18m", rose, white, 20.0)

    add_card("background_checker_plane_1p25m", 2.9, (0.0, 0.0, -1.25), (0.0, 0.0, 0.0), background_mat)
    add_card("near_checker_card_0p75m", 0.62, (-0.42, 0.18, -0.75), (4.0, -8.0, -11.0), near_mat, 0.82)
    add_card("mid_checker_card_0p95m", 0.56, (0.32, 0.10, -0.95), (-5.0, 6.0, 12.0), mid_mat, 1.05)
    add_card("far_checker_card_1p08m", 0.52, (-0.08, -0.29, -1.08), (8.0, 4.0, -6.0), far_mat, 0.9)
    add_card("edge_checker_card_1p18m", 0.46, (0.47, -0.28, -1.18), (-6.0, -4.0, 25.0), edge_mat, 0.72)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.28))
    light = bpy.context.object
    light.name = "front_softbox"
    light.data.energy = 42.0
    light.data.shape = "RECTANGLE"
    light.data.size = 1.6
    light.data.size_y = 1.1

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.10
    scene.world = world

    if ARGS.engine == "cycles":
        backend = configure_cycles(scene, ARGS.device)
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        backend = "EEVEE"
    scene["compute_backend"] = backend
    return camera


def render(camera: bpy.types.Object, path: Path, focus: float | None, fstop: float | None = None) -> None:
    if focus is None:
        camera.data.dof.use_dof = False
    else:
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = focus
        if fstop is not None:
            camera.data.dof.aperture_fstop = fstop
    bpy.context.scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)


def camera_ray(camera: bpy.types.Object, x: int, y: int) -> Vector:
    u = (x + 0.5) / ARGS.width
    v = (y + 0.5) / ARGS.height
    sensor_height = camera.data.sensor_width * ARGS.height / ARGS.width
    half_width = camera.data.sensor_width / (2.0 * camera.data.lens)
    half_height = sensor_height / (2.0 * camera.data.lens)
    local = Vector(((u - 0.5) * 2.0 * half_width, (0.5 - v) * 2.0 * half_height, -1.0))
    return (camera.matrix_world.to_quaternion() @ local).normalized()


def save_depth(camera: bpy.types.Object, output_dir: Path) -> None:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation
    world_to_camera = camera.matrix_world.inverted()
    depth = np.full((ARGS.height, ARGS.width), np.nan, dtype=np.float32)
    object_ids = np.zeros((ARGS.height, ARGS.width), dtype=np.int16)
    names: dict[str, int] = {}

    for y in range(ARGS.height):
        for x in range(ARGS.width):
            hit, location, _normal, _index, obj, _matrix = scene.ray_cast(depsgraph, origin, camera_ray(camera, x, y), distance=5.0)
            if not hit:
                continue
            camera_space = world_to_camera @ location
            depth[y, x] = -float(camera_space.z)
            if obj is not None:
                names.setdefault(obj.name, len(names) + 1)
                object_ids[y, x] = names[obj.name]

    np.save(output_dir / "depth_true_m.npy", depth)
    np.save(output_dir / "object_id.npy", object_ids)
    metadata = {
        "scene": "multi-depth checkerboard cards",
        "resolution": [ARGS.width, ARGS.height],
        "focus_near_m": ARGS.focus_near,
        "focus_far_m": ARGS.focus_far,
        "fstop": ARGS.fstop,
        "samples": ARGS.samples,
        "engine": ARGS.engine,
        "compute_backend": scene["compute_backend"],
        "blender_version": bpy.app.version_string,
        "object_ids": {value: key for key, value in names.items()},
        "note": "depth_true_m.npy is camera-axis metric depth in metres from ray casting.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    camera = build_scene()
    # cam_0 is near-focus and cam_1 is far-focus, matching the slide renderers and the
    # linear_slide_new loader, which reads cam_1 as img_plus (I0) and cam_0 as img_minus (I1).
    render(camera, ARGS.output_dir / "cam_0_near_focus.png", ARGS.focus_near, ARGS.fstop)
    render(camera, ARGS.output_dir / "cam_1_far_focus.png", ARGS.focus_far, ARGS.fstop)
    render(camera, ARGS.output_dir / "rgb_reference.png", None)
    save_depth(camera, ARGS.output_dir)


ARGS = parse_args()
main()
