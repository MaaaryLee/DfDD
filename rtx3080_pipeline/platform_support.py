"""Cross-platform helpers shared by the pipeline orchestrators."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bundled_candidates() -> list[Path]:
    tools = ROOT / "tools"
    if not tools.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(tools.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        for relative in ("blender.exe", "blender", "Blender.app/Contents/MacOS/Blender"):
            candidate = entry / relative
            if candidate.exists():
                found.append(candidate)
    return found


def _installed_candidates() -> list[Path]:
    if sys.platform == "darwin":
        return [
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
            Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        ]
    if sys.platform == "win32":
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        base = program_files / "Blender Foundation"
        if not base.is_dir():
            return []
        return [entry / "blender.exe" for entry in sorted(base.iterdir(), reverse=True) if entry.is_dir()]
    return [Path("/usr/local/bin/blender"), Path("/usr/bin/blender"), Path("/snap/bin/blender")]


def find_blender(explicit: str | None = None) -> str:
    """Locate a Blender executable on Windows, macOS, or Linux.

    Order: explicit argument, then ``BLENDER`` in the environment, then a copy bundled under
    ``tools/``, then ``PATH``, then the platform's default install location.
    """
    if explicit:
        return explicit
    from_env = os.environ.get("BLENDER")
    if from_env:
        return from_env
    for candidate in _bundled_candidates():
        return str(candidate)
    on_path = shutil.which("blender") or shutil.which("Blender")
    if on_path:
        return on_path
    for candidate in _installed_candidates():
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Blender was not found. Pass --blender with its full path, or set the BLENDER "
        "environment variable. On macOS the default is "
        "/Applications/Blender.app/Contents/MacOS/Blender"
    )
