import json
from pathlib import Path

import bpy
from mathutils import Vector

out = Path(__file__).resolve().parent.parent / "outputs" / "hello_room"
scene = bpy.context.scene
camera = bpy.data.objects.get("Hello Room View")
if camera is None:
    camera_data = bpy.data.cameras.new("Hello Room View")
    camera = bpy.data.objects.new("Hello Room View", camera_data)
    scene.collection.objects.link(camera)
camera_data = camera.data
camera.location = (4.45, 12.8, 1.8)
target = Vector((6.0, 13.65, 1.0))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = 18
camera_data.clip_start = 0.05
scene.camera = camera
scene.render.engine = "CYCLES"
scene.render.use_compositing = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.resolution_percentage = 100
scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == "CUDA"
scene.cycles.device = "GPU"
scene.render.filepath = str(out / "hello_room.png")
bpy.ops.render.render(write_still=True)
scene.cycles.samples = 256
scene.render.filepath = "//hello_room.png"
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(out / "hello_room.blend"))
summary_path = out / "scene_info.json"
summary = json.loads(summary_path.read_text())
summary.update({
    "camera": camera.name,
    "preview_camera_position": list(camera.location),
    "preview_camera_target": list(target),
    "preview_camera_lens_mm": camera_data.lens,
    "scene_objects": len(scene.objects),
    "packaged_blender_version": bpy.app.version_string,
})
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
