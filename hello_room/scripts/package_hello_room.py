"""Package the generated scene as a portable, camera-ready Blender file."""

import json
from pathlib import Path

import bpy


workspace = Path(__file__).resolve().parent.parent
out = workspace / "outputs" / "hello_room"
bpy.ops.wm.open_mainfile(filepath=str(out / "coarse" / "scene.blend"))
scene = bpy.context.scene
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.adaptive_threshold = 0.01
scene.render.filepath = "//hello_room.png"
if "unique_assets" in bpy.data.collections:
    for collection in bpy.data.collections["unique_assets"].children:
        collection.hide_viewport = False
if scene.camera is None:
    scene.camera = next(obj for obj in scene.objects if obj.type == "CAMERA")
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
bpy.ops.file.pack_all()
missing = [
    image.name
    for image in bpy.data.images
    if image.source == "FILE"
    and not image.packed_file
    and image.filepath
    and not Path(bpy.path.abspath(image.filepath)).is_file()
]
if missing:
    raise RuntimeError(f"Missing image assets: {missing}")
destination = out / "hello_room.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(destination))
summary = {
    "infinigen_version": "1.19.0",
    "source_commit": "01c39c7f7adcf7363ccbcc57c64410c69f4a4e7c",
    "blender_version": bpy.app.version_string,
    "seed": 0,
    "room": "DiningRoom",
    "configs": ["fast_solve.gin", "singleroom.gin"],
    "terrain_enabled": False,
    "scene_objects": len(scene.objects),
    "mesh_objects": sum(obj.type == "MESH" for obj in scene.objects),
    "materials": len(bpy.data.materials),
    "images": len(bpy.data.images),
    "packed_images": sum(bool(image.packed_file) for image in bpy.data.images),
    "missing_images": missing,
    "camera": scene.camera.name,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "render_samples": 256,
    "denoiser": "OPENIMAGEDENOISE",
    "scene_file": destination.name,
}
(out / "scene_info.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
