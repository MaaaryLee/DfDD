"""Render through Infinigen using the denoiser supported by this WSL setup."""

import runpy

import bpy

from infinigen.core import init


configure_render_cycles = init.configure_render_cycles


def configure_render_with_oidn(*args, **kwargs):
    configure_render_cycles(*args, **kwargs)
    if bpy.context.scene.cycles.use_denoising:
        # CUDA rendering is available here; OptiX is not enumerated by Blender.
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"


init.configure_render_cycles = configure_render_with_oidn
runpy.run_module("infinigen_examples.generate_indoors", run_name="__main__")
