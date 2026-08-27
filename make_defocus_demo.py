"""Build a tiny scene with objects at three depths for testing defocus pairs."""

import bpy
from mathutils import Vector


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    return mat


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 640
scene.render.resolution_y = 400
scene.render.resolution_percentage = 100

for name, location, scale, color in (
    ("Near", (-1.8, 0.0, 0.75), (0.7, 0.7, 0.7), (0.8, 0.08, 0.04)),
    ("Middle", (0.0, 3.0, 0.75), (0.7, 0.7, 0.7), (0.05, 0.5, 0.1)),
    ("Far", (1.8, 7.0, 0.75), (0.7, 0.7, 0.7), (0.04, 0.15, 0.8)),
):
    bpy.ops.mesh.primitive_cube_add(location=location, scale=scale)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(name + "Material", color))

bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 3, 0))
bpy.context.object.data.materials.append(material("GroundMaterial", (0.15, 0.15, 0.15)))

bpy.ops.object.light_add(type="AREA", location=(0, 1, 8))
bpy.context.object.data.energy = 1200
bpy.context.object.data.shape = "DISK"
bpy.context.object.data.size = 8

bpy.ops.object.camera_add(location=(0, -6, 3))
camera = bpy.context.object
target = Vector((0, 3, 0.8))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 50.0
camera.data.sensor_width = 36.0
scene.camera = camera

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
scene.world = world

bpy.ops.wm.save_as_mainfile(filepath="/Users/maaary/Documents/ChatGPT/DfDD/defocus_demo.blend")
