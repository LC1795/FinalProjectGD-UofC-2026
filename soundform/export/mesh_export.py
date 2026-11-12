"""Validate and write OBJ or STL meshes."""

from __future__ import annotations

from pathlib import Path

import trimesh

from soundform.errors import ExportError
from soundform.geometry.mesh_utils import validate_mesh


def export_mesh(mesh: trimesh.Trimesh, path: str) -> dict[str, float | int | str]:
    validate_mesh(mesh)
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".obj", ".stl"}:
        raise ExportError("Export geometry as .obj or .stl.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(destination)
    except Exception as exc:
        raise ExportError(f"The mesh could not be written. ({exc})") from exc
    if not destination.is_file() or destination.stat().st_size < 32:
        raise ExportError("The export file is missing or empty.")
    try:
        loaded = trimesh.load(destination, force="mesh", process=False)
    except Exception as exc:
        raise ExportError(f"The file was written but could not be read back. ({exc})") from exc
    if loaded.faces is None or len(loaded.faces) == 0 or len(loaded.vertices) == 0:
        raise ExportError("The exported file does not contain a triangle mesh.")
    return {
        "path": str(destination),
        "vertices": int(len(loaded.vertices)),
        "faces": int(len(loaded.faces)),
        "bytes": int(destination.stat().st_size),
    }
