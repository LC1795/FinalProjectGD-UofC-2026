"""Protective terrain: a continuous surface that rises toward persistent sound.

The height at a plan point comes from the horizontal score of the direction
that runs from the occupant to that point. After smoothing, the protected
zone is forced back to the minimum height so the occupant is not buried.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from soundform.acoustics.response import ResponseField
from soundform.geometry.design_space import (
    design_xy_to_sound_azimuth,
    height_from_score,
    sample_azimuth,
)
from soundform.geometry.mesh_utils import as_mesh
from soundform.models.project import DesignParameters


def build_terrain(response: ResponseField, params: DesignParameters):
    resolution = int(np.clip(params.grid_resolution, 16, 160))
    if params.site_shape == "rectangle":
        half_x = params.site_width * 0.5
        half_y = params.site_depth * 0.5
    else:
        half_x = half_y = float(params.site_radius)
    xs = np.linspace(-half_x, half_x, resolution)
    ys = np.linspace(-half_y, half_y, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    dx = grid_x - params.protected_center_x
    dy = grid_y - params.protected_center_y
    sound_azimuth = design_xy_to_sound_azimuth(dx, dy, params.rotation_offset_deg)
    score = sample_azimuth(sound_azimuth, response.azimuth_deg, response.azimuth_score)
    heights = height_from_score(score, params)

    if params.site_shape == "circle":
        outside = grid_x**2 + grid_y**2 > params.site_radius**2
        heights = heights.copy()
        heights[outside] = params.min_height

    sigma = float(params.smoothing) * 2.2
    if sigma > 0.05:
        heights = gaussian_filter(heights, sigma=sigma, mode="nearest")

    distance = np.hypot(dx, dy)
    margin = max(1.5, float(params.protected_radius) * 0.45)
    # 0 inside the zone, 1 outside the margin. Applied after smoothing.
    blend = np.clip((distance - params.protected_radius) / margin, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    heights = params.min_height + (heights - params.min_height) * blend

    index = -np.ones((resolution, resolution), dtype=np.int64)
    vertices: list[tuple[float, float, float]] = []
    for iy in range(resolution):
        for ix in range(resolution):
            if params.site_shape == "circle" and xs[ix] ** 2 + ys[iy] ** 2 > params.site_radius**2 + 1e-8:
                continue
            index[iy, ix] = len(vertices)
            vertices.append((float(xs[ix]), float(ys[iy]), float(heights[iy, ix])))

    faces: list[tuple[int, int, int]] = []
    for iy in range(resolution - 1):
        for ix in range(resolution - 1):
            a = index[iy, ix]
            b = index[iy, ix + 1]
            c = index[iy + 1, ix + 1]
            d = index[iy + 1, ix]
            if min(a, b, c, d) < 0:
                continue
            faces.append((int(a), int(b), int(c)))
            faces.append((int(a), int(c), int(d)))
    if params.site_shape == "circle":
        _project_rim(index, vertices, float(params.site_radius))
    return as_mesh(np.asarray(vertices), np.asarray(faces, dtype=np.int64))


def _project_rim(index: np.ndarray, vertices: list[tuple[float, float, float]], radius: float) -> None:
    """Pull staircase edge vertices onto the site circle."""
    height, width = index.shape
    for iy in range(height):
        for ix in range(width):
            vertex = index[iy, ix]
            if vertex < 0:
                continue
            on_rim = False
            for ny, nx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width or index[ny, nx] < 0:
                    on_rim = True
                    break
            if not on_rim:
                continue
            x, y, z = vertices[vertex]
            length = float(np.hypot(x, y))
            if length <= 1e-8:
                continue
            vertices[vertex] = (x / length * radius, y / length * radius, z)
