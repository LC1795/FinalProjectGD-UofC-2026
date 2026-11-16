"""Section-based mesh operations for model-authored pavilion recipes."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from soundform.geometry.mesh_utils import as_mesh


def _triangulate_cap(loop):
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator

    points, polygon, triangles = vtkPoints(), vtkIdList(), vtkCellArray()
    points.SetDataTypeToDouble()
    for index, point in enumerate(loop):
        points.InsertNextPoint(*point)
        polygon.InsertNextId(index)
    if not vtkContourTriangulator.TriangulatePolygon(polygon, points, triangles):
        raise ValueError("Could not triangulate the cap. Use a simple closed loop without crossings.")
    ids = vtkIdList()
    faces = []
    triangles.InitTraversal()
    while triangles.GetNextCell(ids):
        faces.append(tuple(ids.GetId(index) for index in range(3)))
    return faces


def _skin(sections: np.ndarray, cap: bool, closed_profile: bool = True):
    """Join matching section loops, preserving their vertex correspondence."""
    count, sides, _ = sections.shape
    faces = []
    for i in range(count - 1):
        start = i * sides
        for j in range(sides if closed_profile else sides - 1):
            a, b = start + j, start + (j + 1) % sides
            faces.extend(((a, b, b + sides), (a, b + sides, a + sides)))
    vertices = list(sections.reshape(-1, 3))
    if cap:
        last = (count - 1) * sides
        for offset, reverse in ((0, True), (last, False)):
            loop = sections[offset // sides]
            center = loop.mean(axis=0)
            centered = loop - center
            normal = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
            norm = np.linalg.norm(normal)
            if norm < 1e-9 or np.max(np.abs(centered @ (normal / norm))) > 1e-5:
                raise ValueError("Capped sections need planar loops with nonzero area.")
            edges = np.roll(loop, -1, axis=0) - loop
            turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ (normal / norm)
            if np.min(turns) < -1e-10:
                cap_faces = [tuple(offset + index for index in face) for face in _triangulate_cap(loop)]
            else:
                center_index = len(vertices)
                vertices.append(center)
                cap_faces = [(center_index, offset + j, offset + (j + 1) % sides) for j in range(sides)]
            faces.extend(face[::-1] if reverse else face for face in cap_faces)
    mesh = as_mesh(np.asarray(vertices), np.asarray(faces))
    mesh.fix_normals()
    if cap and (not mesh.is_watertight or mesh.volume < 1e-9):
        raise ValueError("The capped sections did not form a closed volume.")
    return mesh


def _frames(path: np.ndarray):
    """Carry a local profile frame along the rail using minimal rotation."""
    tangents = np.gradient(path, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("The sweep rail has a stationary or reversing segment.")
    tangents /= lengths[:, None]
    tangent = tangents[0]
    reference = np.array([0., 0., 1.]) if abs(tangent[2]) < 0.95 else np.array([1., 0., 0.])
    u = np.cross(reference, tangent)
    u /= np.linalg.norm(u)
    axes = []
    for current in tangents:
        cross = np.cross(tangent, current)
        dot = float(tangent @ current)
        if dot < -0.9999:
            raise ValueError("The sweep rail reverses direction. Use separate rails for sharp forks.")
        u = u + np.cross(cross, u) + np.cross(cross, np.cross(cross, u)) / max(1 + dot, 1e-8)
        u -= (u @ current) * current
        u /= np.linalg.norm(u)
        axes.append((u.copy(), np.cross(current, u)))
        tangent = current
    return np.asarray(axes)


def _sweep(rail, profile, scales):
    distances = np.linalg.norm(np.diff(rail, axis=0), axis=1)
    if np.any(distances < 1e-6):
        raise ValueError("Consecutive rail points must be distinct.")
    if len(scales) != len(rail) or np.min(scales) <= 0:
        raise ValueError("Provide one positive scale or radius for each rail point.")
    knots = np.r_[0.0, distances.cumsum()]
    samples = np.linspace(0, knots[-1], 1 + 3 * len(distances))
    path = PchipInterpolator(knots, rail, axis=0)(samples)
    scale = np.interp(samples, knots, scales)
    frame = _frames(path)
    sections = path[:, None, :] + np.einsum("vc,rcd->rvd", profile, frame) * scale[:, None, None]
    return _skin(sections, cap=True)


def build_section_shape(shape, geometry):
    if shape == "loft":
        if geometry["cap"] not in (0, 1):
            raise ValueError("Loft cap must be 0 (open) or 1 (closed).")
        closed = geometry.get("closed_profile", 1)
        if closed not in (0, 1) or (not closed and geometry["cap"]):
            raise ValueError("An open loft profile requires closed_profile=0 and cap=0.")
        return _skin(geometry["sections"], bool(geometry["cap"]), bool(closed))
    rail = geometry["rail"]
    if shape == "pipe":
        if np.min(geometry["radii"]) < 0.005:
            raise ValueError("Pipe radii must be at least 5 mm.")
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        profile = np.column_stack((np.cos(angles), np.sin(angles)))
        scales = geometry["radii"]
    else:
        profile, scales = geometry["profile"], geometry["scales"]
    return _sweep(rail, profile, scales)
