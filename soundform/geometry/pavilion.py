"""Protective pavilion: floor, directionally scaled walls, and a roof.

Loud, persistent directions get taller enclosure. The quietest direction is
left as an opening of the requested width, with a lintel so the roof stays
connected. Inside the protected radius the roof is held up at the crown so
the occupant is not buried in the shell.
"""

from __future__ import annotations

import numpy as np

from soundform.acoustics.response import ResponseField
from soundform.errors import GeometryError
from soundform.geometry.design_space import (
    design_xy_to_sound_azimuth,
    sample_azimuth,
)
from soundform.geometry.mesh_utils import as_mesh
from soundform.models.project import DesignParameters


def build_pavilion(response: ResponseField, params: DesignParameters, warnings: list[str]):
    samples = 72
    outer = max(1.0, float(params.pavilion_diameter) * 0.5)
    thickness = float(np.clip(params.wall_thickness, 0.08, outer * 0.45))
    inner = outer - thickness
    if inner <= params.protected_radius + 0.4:
        raise GeometryError(
            "The pavilion's inner face cuts into the protected zone. "
            "Increase the pavilion diameter or reduce wall thickness or the protected radius."
        )
    angles = (np.arange(samples) + 0.5) * (360.0 / samples)
    radians = np.deg2rad(angles)
    sound_azimuth = design_xy_to_sound_azimuth(-np.sin(radians), np.cos(radians), params.rotation_offset_deg)
    score = sample_azimuth(sound_azimuth, response.azimuth_deg, response.azimuth_score)
    peak = float(np.max(score)) if score.size else 0.0
    if peak > 1e-8:
        score = score / peak
    enclosure = float(np.clip(params.enclosure, 0.0, 1.0))
    directional = np.clip(params.response_intensity, 0.0, 3.0) * score
    presence = np.clip((1.0 - enclosure) * directional + enclosure * np.maximum(directional, 0.35), 0.0, 1.0)
    span = max(0.0, params.max_height - params.min_height)
    wall_height = params.min_height + presence * span

    opening = float(np.clip(params.opening_width_deg, 0.0, 160.0))
    quietest = float(angles[int(np.argmin(score))])
    delta = np.abs(((angles - quietest + 180.0) % 360.0) - 180.0)
    opening_mask = delta <= opening * 0.5
    solid_height = wall_height.copy()
    solid_height[opening_mask] = 0.0
    positive = wall_height[~opening_mask]
    lintel = float(np.median(positive) * 0.55) if positive.size else params.min_height
    lintel = float(np.clip(lintel, 0.0, params.max_height))
    roof_edge = solid_height.copy()
    roof_edge[opening_mask] = lintel

    curvature = float(np.clip(params.roof_curvature, 0.0, 1.0))
    crown = (1.0 - curvature) * float(np.max(roof_edge)) + curvature * float(params.max_height)
    crown = max(crown, float(np.max(roof_edge)))
    if params.max_height < 2.2:
        warnings.append(
            "Maximum height is below 2.2 m, so the pavilion does not leave standing clearance."
        )

    cx = float(params.protected_center_x)
    cy = float(params.protected_center_y)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_ring(radius: float, heights: np.ndarray) -> int:
        start = len(vertices)
        radians_local = np.deg2rad(angles)
        for index, azimuth in enumerate(radians_local):
            vertices.append(
                (
                    cx - np.sin(azimuth) * radius,
                    cy + np.cos(azimuth) * radius,
                    float(heights[index]),
                )
            )
        return start

    floor_center = len(vertices)
    vertices.append((cx, cy, 0.0))
    floor_ring = add_ring(outer, np.zeros(samples))
    for index in range(samples):
        faces.append((floor_center, floor_ring + index, floor_ring + (index + 1) % samples))

    inner_bottom = add_ring(inner, np.zeros(samples))
    outer_bottom = add_ring(outer, np.zeros(samples))
    inner_top = add_ring(inner, roof_edge)
    outer_top = add_ring(outer, roof_edge)

    for index in range(samples):
        nxt = (index + 1) % samples
        if opening_mask[index]:
            # Lintel only: the opening stays walkable and the roof stays tied to the walls.
            faces.append((inner_top + index, outer_top + index, outer_top + nxt))
            faces.append((inner_top + index, outer_top + nxt, inner_top + nxt))
            continue
        faces.append((outer_bottom + index, outer_bottom + nxt, outer_top + nxt))
        faces.append((outer_bottom + index, outer_top + nxt, outer_top + index))
        faces.append((inner_bottom + index, inner_top + index, inner_top + nxt))
        faces.append((inner_bottom + index, inner_top + nxt, inner_bottom + nxt))
        faces.append((inner_top + index, outer_top + index, outer_top + nxt))
        faces.append((inner_top + index, outer_top + nxt, inner_top + nxt))
        faces.append((inner_bottom + index, inner_bottom + nxt, outer_bottom + nxt))
        faces.append((inner_bottom + index, outer_bottom + nxt, outer_bottom + index))

    # Roof: hold the crown flat over the protected zone, then descend to the wall.
    clear_s = float(np.clip(params.protected_radius / inner, 0.05, 0.85))
    rings = 7
    roof_ids = []
    crown_index = len(vertices)
    vertices.append((cx, cy, crown))
    for ring in range(1, rings + 1):
        fraction = ring / rings
        if fraction <= clear_s:
            height = crown
        else:
            blend = (fraction - clear_s) / (1.0 - clear_s)
            blend = blend * blend * (3.0 - 2.0 * blend)
            height_ring = crown * (1.0 - blend) + roof_edge * blend
            height = height_ring
        roof_ids.append(add_ring(inner * fraction, np.full(samples, 1.0) * height if np.isscalar(height) else height))

    first = roof_ids[0]
    for index in range(samples):
        faces.append((crown_index, first + (index + 1) % samples, first + index))
    for ring in range(len(roof_ids) - 1):
        inner_ring = roof_ids[ring]
        outer_ring = roof_ids[ring + 1]
        for index in range(samples):
            nxt = (index + 1) % samples
            faces.append((inner_ring + index, inner_ring + nxt, outer_ring + nxt))
            faces.append((inner_ring + index, outer_ring + nxt, outer_ring + index))
    # Stitch the outer roof ring to the inner wall top. They coincide in radius
    # but are separate vertices; welding happens in merge_vertices.
    outer_roof = roof_ids[-1]
    for index in range(samples):
        nxt = (index + 1) % samples
        faces.append((outer_roof + index, outer_roof + nxt, inner_top + nxt))
        faces.append((outer_roof + index, inner_top + nxt, inner_top + index))

    mesh = as_mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh
