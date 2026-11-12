"""Coordinate frame of the abstract design environment.

The recording is not placed into the room where it was captured. The microphone
is co-located with the occupant zone because a first-order recording has a
direction at the capsule and no source distance. Design +Y is "forward" when
the rotation offset is 0, which matches microphone forward (sound +X).

Positive rotation offset turns the decoded field counter-clockwise about up,
as viewed from above.
"""

from __future__ import annotations

import numpy as np

from soundform.errors import GeometryError
from soundform.models.project import DesignParameters

EAR_HEIGHT_M = 1.5


def design_xy_to_sound_azimuth(dx: np.ndarray, dy: np.ndarray, rotation_ccw_deg: float) -> np.ndarray:
    """Sound-field azimuth, in degrees, of a horizontal design-space offset."""
    angle = np.deg2rad(-rotation_ccw_deg)
    unrotated_x = np.cos(angle) * dx - np.sin(angle) * dy
    unrotated_y = np.sin(angle) * dx + np.cos(angle) * dy
    return np.mod(np.degrees(np.arctan2(-unrotated_x, unrotated_y)), 360.0)


def sound_az_el_to_design_vector(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    rotation_ccw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a microphone direction to a unit vector in the design frame."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    forward = np.cos(elevation) * np.cos(azimuth)
    left = np.cos(elevation) * np.sin(azimuth)
    up = np.sin(elevation)
    design_x = -left
    design_y = forward
    turn = np.deg2rad(rotation_ccw_deg)
    rotated_x = np.cos(turn) * design_x - np.sin(turn) * design_y
    rotated_y = np.sin(turn) * design_x + np.cos(turn) * design_y
    return rotated_x, rotated_y, up


def sample_azimuth(azimuth_deg: np.ndarray, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a uniform azimuth array. `centers` start at half a bin."""
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if centers.size == 0:
        return np.zeros_like(azimuth_deg, dtype=np.float64)
    step = 360.0 / centers.size
    position = np.mod(np.asarray(azimuth_deg, dtype=np.float64), 360.0) / step - 0.5
    index0 = np.floor(position).astype(int) % centers.size
    index1 = (index0 + 1) % centers.size
    blend = position - np.floor(position)
    return (1.0 - blend) * values[index0] + blend * values[index1]


def characteristic_radius(params: DesignParameters) -> float:
    if params.site_shape == "rectangle":
        return 0.5 * float(min(params.site_width, params.site_depth))
    return float(params.site_radius)


def prepare_parameters(params: DesignParameters) -> tuple[DesignParameters, list[str]]:
    """Validate dimensions and keep the occupant zone inside the site."""
    used = params.copy()
    warnings: list[str] = []
    if used.max_height < used.min_height:
        raise GeometryError("Maximum height must be at least the minimum height.")
    if used.protected_radius <= 0:
        raise GeometryError("The protected-zone radius must be greater than zero.")
    if used.site_shape == "circle":
        if used.site_radius <= used.protected_radius + 0.5:
            raise GeometryError(
                "The site radius must leave room outside the protected zone. "
                "Increase the site or reduce the protected radius."
            )
        limit = used.site_radius - used.protected_radius
        offset = float(np.hypot(used.protected_center_x, used.protected_center_y))
        if offset > limit and offset > 0:
            scale = limit / offset
            used.protected_center_x *= scale
            used.protected_center_y *= scale
            warnings.append(
                "The occupant zone was moved inward so it stays inside the circular site."
            )
    elif used.site_shape == "rectangle":
        if used.site_width <= 2 * used.protected_radius + 1 or used.site_depth <= 2 * used.protected_radius + 1:
            raise GeometryError(
                "The rectangular site must be larger than the protected zone. "
                "Increase the site width and depth or reduce the protected radius."
            )
        limit_x = used.site_width * 0.5 - used.protected_radius
        limit_y = used.site_depth * 0.5 - used.protected_radius
        clamped_x = float(np.clip(used.protected_center_x, -limit_x, limit_x))
        clamped_y = float(np.clip(used.protected_center_y, -limit_y, limit_y))
        if abs(clamped_x - used.protected_center_x) > 1e-6 or abs(clamped_y - used.protected_center_y) > 1e-6:
            used.protected_center_x = clamped_x
            used.protected_center_y = clamped_y
            warnings.append("The occupant zone was moved so it stays inside the rectangular site.")
    else:
        raise GeometryError("The site shape must be circular or rectangular.")
    if used.grid_resolution < 16:
        raise GeometryError("Grid resolution must be at least 16.")
    used.smoothing = float(np.clip(used.smoothing, 0.0, 1.0))
    used.response_intensity = float(max(0.0, used.response_intensity))
    return used, warnings


def height_from_score(score: np.ndarray, params: DesignParameters) -> np.ndarray:
    span = max(0.0, float(params.max_height - params.min_height))
    gain = np.clip(float(params.response_intensity) * np.asarray(score, dtype=np.float64), 0.0, 1.0)
    return float(params.min_height) + gain * span


def contains_plan(x: np.ndarray, y: np.ndarray, params: DesignParameters) -> np.ndarray:
    if params.site_shape == "rectangle":
        return (np.abs(x) <= params.site_width * 0.5) & (np.abs(y) <= params.site_depth * 0.5)
    return (x * x + y * y) <= params.site_radius**2
