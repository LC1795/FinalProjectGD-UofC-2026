"""Directional barrier walls.

Segments are placed on a ring around the occupant. Height follows the
horizontal score, so a loud direction gets a taller barrier than a quiet one.
Openness drops the quietest fraction of the ring, and an entry is always cut
at the quietest direction. This is a geometric arrangement, not a prediction
of transmission loss.
"""

from __future__ import annotations

import numpy as np

from soundform.acoustics.response import ResponseField
from soundform.geometry.design_space import (
    design_xy_to_sound_azimuth,
    height_from_score,
    sample_azimuth,
)
from soundform.geometry.mesh_utils import as_mesh, quad
from soundform.models.project import DesignParameters


def _arc(cx: float, cy: float, radius: float, azimuths: np.ndarray, z: float) -> np.ndarray:
    radians = np.deg2rad(azimuths)
    return np.column_stack(
        (
            cx - np.sin(radians) * radius,
            cy + np.cos(radians) * radius,
            np.full(azimuths.shape, z, dtype=np.float64),
        )
    )


def _closed_segment(vertices: list, faces: list, params: DesignParameters, az0: float, az1: float, height: float, radius: float) -> None:
    curvature = float(np.clip(params.curvature, 0.0, 1.0))
    count = 2 + int(round(curvature * 6))
    azimuths = np.linspace(az0, az1, count)
    inner = radius
    outer = radius + max(0.05, float(params.wall_thickness))
    base = 0.0
    top = max(base, height)
    rings = [
        _arc(params.protected_center_x, params.protected_center_y, inner, azimuths, base),
        _arc(params.protected_center_x, params.protected_center_y, outer, azimuths, base),
        _arc(params.protected_center_x, params.protected_center_y, inner, azimuths, top),
        _arc(params.protected_center_x, params.protected_center_y, outer, azimuths, top),
    ]
    start = len(vertices)
    for ring in rings:
        vertices.extend(map(tuple, ring))
    n = count
    inner_bottom = start
    outer_bottom = start + n
    inner_top = start + 2 * n
    outer_top = start + 3 * n
    for i in range(n - 1):
        faces.extend(quad(outer_bottom + i, outer_bottom + i + 1, outer_top + i + 1, outer_top + i))
        faces.extend(quad(inner_bottom + i, inner_top + i, inner_top + i + 1, inner_bottom + i + 1))
        faces.extend(quad(inner_top + i, outer_top + i, outer_top + i + 1, inner_top + i + 1))
        faces.extend(quad(inner_bottom + i, inner_bottom + i + 1, outer_bottom + i + 1, outer_bottom + i))
    faces.extend(quad(inner_bottom, outer_bottom, outer_top, inner_top))
    last = n - 1
    faces.extend(
        quad(
            inner_bottom + last,
            inner_top + last,
            outer_top + last,
            outer_bottom + last,
        )
    )


def build_walls(response: ResponseField, params: DesignParameters):
    segment_count = 36
    width = 360.0 / segment_count
    gap = min(2.0, width * 0.15)
    centers = (np.arange(segment_count) + 0.5) * width
    radians = np.deg2rad(centers)
    dx = -np.sin(radians)
    dy = np.cos(radians)
    sound_azimuth = design_xy_to_sound_azimuth(dx, dy, params.rotation_offset_deg)
    scores = sample_azimuth(sound_azimuth, response.azimuth_deg, response.azimuth_score)
    openness = float(np.clip(params.openness, 0.0, 1.0))
    if openness <= 1e-6 or np.allclose(scores, scores[0]):
        active = np.ones(segment_count, dtype=bool)
    else:
        threshold = float(np.quantile(scores, openness))
        active = scores >= threshold
    quietest = int(np.argmin(scores))
    entry = float(np.clip(params.entry_width_deg, 0.0, 180.0))
    for index, center in enumerate(centers):
        delta = abs(((center - centers[quietest] + 180.0) % 360.0) - 180.0)
        if delta <= entry * 0.5:
            active[index] = False
    if not np.any(active):
        # Keep the loudest segment so the mode still answers the field.
        active[int(np.argmax(scores))] = True

    radius = float(params.protected_radius + max(0.2, params.barrier_spacing))
    heights = height_from_score(scores, params)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for index in range(segment_count):
        if not active[index] or float(heights[index]) <= 1e-3:
            continue
        az0 = centers[index] - width * 0.5 + gap * 0.5
        az1 = centers[index] + width * 0.5 - gap * 0.5
        _closed_segment(vertices, faces, params, az0, az1, float(heights[index]), radius)
    ground = _ground_disc(params)
    from soundform.geometry.mesh_utils import merge_meshes

    if not faces:
        return ground
    wall_mesh = as_mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    return merge_meshes([ground, wall_mesh])


def _ground_disc(params: DesignParameters, samples: int = 64):
    """Flat base at z = 0. It is ground, not a barrier."""
    if params.site_shape == "rectangle":
        half_x = params.site_width * 0.5
        half_y = params.site_depth * 0.5
        vertices = np.array(
            [
                [-half_x, -half_y, 0.0],
                [half_x, -half_y, 0.0],
                [half_x, half_y, 0.0],
                [-half_x, half_y, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        return as_mesh(vertices, faces)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    ring = np.column_stack(
        (
            np.cos(angles) * params.site_radius,
            np.sin(angles) * params.site_radius,
            np.zeros(samples),
        )
    )
    vertices = np.vstack([np.zeros((1, 3)), ring])
    faces = []
    for index in range(samples):
        faces.append((0, 1 + index, 1 + (index + 1) % samples))
    return as_mesh(vertices, np.asarray(faces, dtype=np.int64))
