"""Generate the Infinigen-demo-style brick torus using Blender only.

No Infinigen or third-party Python packages are required. Run with Blender:

    blender --background --python make_brick_torus_standalone.py -- \
        --output-dir outputs/brick_torus --engine cycles
"""

import argparse
import math
import sys
from pathlib import Path

import bpy


def args_after_double_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/brick_torus"))
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(args_after_double_dash())


def brick_material():
    material = bpy.data.materials.new("Procedural_Bricks")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    brick = nodes.new("ShaderNodeTexBrick")
    bump = nodes.new("ShaderNodeBump")
    texcoord = nodes.new("ShaderNodeTexCoord")

    brick.inputs["Color1"].default_value = (0.32, 0.055, 0.025, 1.0)
    brick.inputs["Color2"].default_value = (0.62, 0.20, 0.10, 1.0)
    brick.inputs["Mortar"].default_value = (0.72, 0.68, 0.61, 1.0)
    brick.inputs["Scale"].default_value = 7.0
    brick.inputs["Mortar Size"].default_value = 0.035
    brick.inputs["Mortar Smooth"].default_value = 0.02
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.squash = 1.0

    shader.inputs["Roughness"].default_value = 0.72
    bump.inputs["Strength"].default_value = 0.45
    bump.inputs["Distance"].default_value = 0.12

    links.new(texcoord.outputs["Generated"], brick.inputs["Vector"])
    links.new(brick.outputs["Color"], shader.inputs["Base Color"])
    links.new(brick.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def checker_material():
    material = bpy.data.materials.new("Reference_Grid")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    checker = nodes.new("ShaderNodeTexChecker")
    coord = nodes.new("ShaderNodeTexCoord")
    checker.inputs["Color1"].default_value = (0.72, 0.72, 0.68, 1.0)
    checker.inputs["Color2"].default_value = (0.92, 0.90, 0.84, 1.0)
    checker.inputs["Scale"].default_value = 24.0
    shader.inputs["Roughness"].default_value = 0.85
    links.new(coord.outputs["Generated"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], shader.inputs["Base Color"])
    return material


def plain_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.65
    return material


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"

    # Geometry parameters from Infinigen 2's material_torus_uv demo.
    bpy.ops.mesh.primitive_torus_add(
        align="WORLD",
        major_segments=256,
        minor_segments=128,
        location=(0.0, 0.0, 0.0),
        rotation=tuple(math.radians(v) for v in (-40.0, -25.0, 140.0)),
        major_radius=0.5,
        minor_radius=0.25,
    )
    torus = bpy.context.object
    torus.name = "Brick_Torus"
    torus.location.z = torus.dimensions.z / 2.0
    torus.data.materials.append(brick_material())
    for polygon in torus.data.polygons:
        polygon.use_smooth = True

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "Checker_Floor"
    floor.data.materials.append(checker_material())

    # The white reference object shown beside the torus in the documentation image.
    height = 1.65
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=0.3,
        depth=height,
        location=(0.38, 0.76, height / 2.0 - 0.10),
    )
    reference = bpy.context.object
    reference.name = "Scale_Reference"
    reference.data.materials.append(plain_material("Reference_White", (0.9, 0.9, 0.88)))
    bevel = reference.modifiers.new("Soft edges", "BEVEL")
    bevel.width = 0.15
    bevel.segments = 8

    # Match the Infinigen demo's hard-coded camera placement.
    bpy.ops.object.camera_add(
        location=(3.25, -2.60, torus.location.z + 1.43),
        rotation=tuple(math.radians(v) for v in (71.0, 0.0, 52.0)),
    )
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.lens = 50.0
    scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(-2.0, -2.5, 5.0))
    key = bpy.context.object
    key.name = "Key_Light"
    key.data.energy = 900.0
    key.data.shape = "DISK"
    key.data.size = 4.0

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.82, 0.78, 0.68, 1.0)
    background.inputs["Strength"].default_value = 0.45
    scene.world = world
    return scene


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args)
    scene.render.filepath = str(output_dir / "brick_torus.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "brick_torus.blend"))
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {output_dir / 'brick_torus.blend'}")
    print(f"Wrote {output_dir / 'brick_torus.png'}")


main()
