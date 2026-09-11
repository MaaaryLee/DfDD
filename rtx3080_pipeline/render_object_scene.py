"""Render a shaped synthetic scene for DfDD/monocular-depth comparison.

This is intentionally separate from ``render_linear_slide.py``. The linear
slide data uses a flat target for calibration; this scene contains objects at
multiple depths so dense depth maps have visible shape.
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


def procedural_material(name: str, base: tuple[float, float, float, float], scale: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")

    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.62
    color_ramp.color_ramp.elements[0].position = 0.22
    color_ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = base
    shader.inputs["Roughness"].default_value = 0.78

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], shader.inputs["Base Color"])
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


def build_scene() -> bpy.types.Object:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = ARGS.width
    scene.render.resolution_y = ARGS.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    look_items = {item.identifier for item in scene.view_settings.bl_rna.properties["look"].enum_items}
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "AgX - Base Contrast", "None"):
        if look in look_items:
            scene.view_settings.look = look
            break

    camera = add_camera(scene)

    mat_floor = procedural_material("mat_background", (0.80, 0.72, 0.22, 1.0), 42.0)
    mat_cube = procedural_material("mat_cube", (0.96, 0.22, 0.20, 1.0), 55.0)
    mat_sphere = procedural_material("mat_sphere", (0.18, 0.56, 0.95, 1.0), 65.0)
    mat_cylinder = procedural_material("mat_cylinder", (0.15, 0.78, 0.42, 1.0), 75.0)
    mat_cone = procedural_material("mat_cone", (0.80, 0.42, 0.94, 1.0), 68.0)

    bpy.ops.mesh.primitive_plane_add(size=2.8, location=(0.0, 0.0, -1.25))
    background = bpy.context.object
    background.name = "background_plane_1p25m"
    background.data.materials.append(mat_floor)

    bpy.ops.mesh.primitive_cube_add(size=0.30, location=(-0.38, 0.16, -0.82))
    cube = bpy.context.object
    cube.name = "cube_near_0p82m"
    cube.rotation_euler = (0.18, 0.34, -0.18)
    cube.data.materials.append(mat_cube)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.19, location=(0.28, 0.05, -0.98))
    sphere = bpy.context.object
    sphere.name = "sphere_mid_0p98m"
    sphere.data.materials.append(mat_sphere)

    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.13, depth=0.28, location=(-0.02, -0.28, -1.08))
    cylinder = bpy.context.object
    cylinder.name = "cylinder_far_1p08m"
    cylinder.rotation_euler = (math.radians(90), 0.0, math.radians(12))
    cylinder.data.materials.append(mat_cylinder)

    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.16, radius2=0.02, depth=0.34, location=(0.43, -0.27, -1.15))
    cone = bpy.context.object
    cone.name = "cone_far_1p15m"
    cone.rotation_euler = (math.radians(90), 0.0, math.radians(-15))
    cone.data.materials.append(mat_cone)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.25))
    front_light = bpy.context.object
    front_light.name = "front_softbox"
    front_light.data.energy = 55.0
    front_light.data.shape = "RECTANGLE"
    front_light.data.size = 1.8
    front_light.data.size_y = 1.2

    bpy.ops.object.light_add(type="POINT", location=(-0.85, 0.55, -0.55))
    side_light = bpy.context.object
    side_light.name = "side_highlight"
    side_light.data.energy = 18.0

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
    width = ARGS.width
    height = ARGS.height
    u = (x + 0.5) / width
    v = (y + 0.5) / height
    sensor_height = camera.data.sensor_width * height / width
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
