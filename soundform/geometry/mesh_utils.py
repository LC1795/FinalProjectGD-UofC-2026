"""Triangle-mesh helpers shared by the form generators and the exporter."""

from __future__ import annotations

import numpy as np
import trimesh

from soundform.errors import ExportError, GeometryError


def as_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    if len(vertices) < 3 or len(faces) < 1:
        raise GeometryError("The generator produced an empty mesh. Adjust the site or the response.")
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise GeometryError("The mesh collapsed after removing degenerate faces.")
    return mesh


def merge_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    populated = [mesh for mesh in meshes if mesh is not None and len(mesh.faces)]
    if not populated:
        raise GeometryError("There is no geometry to combine.")
    if len(populated) == 1:
        return populated[0]
    combined = trimesh.util.concatenate(populated)
    combined.merge_vertices()
    combined.update_faces(combined.nondegenerate_faces())
    combined.remove_unreferenced_vertices()
    return combined


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    """Reject meshes that cannot be exported as OBJ or STL."""
    if mesh is None or mesh.vertices is None or mesh.faces is None:
        raise ExportError("There is no mesh to export. Generate architecture first.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ExportError("The mesh has no vertices or faces.")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ExportError("The mesh faces are not triangles.")
    if not np.isfinite(mesh.vertices).all():
        raise ExportError("The mesh contains NaN or infinite vertices.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ExportError("The mesh faces refer to vertices that do not exist.")
    areas = mesh.area_faces
    if areas.size == 0 or float(np.max(areas)) <= 0.0:
        raise ExportError("The mesh has no surface area.")


def quad(a: int, b: int, c: int, d: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (a, b, c), (a, c, d)
