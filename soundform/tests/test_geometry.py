import numpy as np
import pytest

from soundform.acoustics.protection_estimator import estimate_protection
from soundform.acoustics.response import compute_response
from soundform.geometry.design_space import design_xy_to_sound_azimuth
from soundform.geometry.generate import generate_architecture
from soundform.models.project import DesignParameters
from soundform.tests.helpers import angular_error_deg, blob_field


def _params(**overrides) -> DesignParameters:
    params = DesignParameters(
        site_radius=24.0,
        site_width=48.0,
        site_depth=40.0,
        max_height=8.0,
        min_height=0.0,
        grid_resolution=36,
        smoothing=0.0,
        response_intensity=1.0,
        protected_radius=4.0,
        wall_thickness=0.4,
        barrier_spacing=2.5,
        curvature=0.8,
        openness=0.45,
        entry_width_deg=30.0,
        pavilion_diameter=16.0,
        roof_curvature=0.4,
        enclosure=0.2,
        opening_width_deg=36.0,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def _weights_on(field):
    return compute_response(
        field,
        {
            "broadband": 1.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "persistence": 0.0,
            "transient": 0.0,
            "confidence": 0.0,
        },
    )


def _sector_height(mesh, center, target_azimuth, width=35.0):
    points = mesh.vertices
    offset_x = points[:, 0] - center[0]
    offset_y = points[:, 1] - center[1]
    azimuth = design_xy_to_sound_azimuth(offset_x, offset_y, 0.0)
    delta = np.abs(((azimuth - target_azimuth + 180.0) % 360.0) - 180.0)
    chosen = points[delta < width]
    if chosen.size == 0:
        return 0.0
    return float(np.max(chosen[:, 2]))


def test_terrain_rises_toward_the_sound_and_spares_the_occupant():
    field = blob_field(0.0, 0.0, spread_deg=12.0)
    mesh, _response, used, _warnings = generate_architecture(field, _params(mode="terrain"))
    assert np.isfinite(mesh.vertices).all()
    assert len(mesh.faces) > 10
    radius = np.hypot(mesh.vertices[:, 0] - used.protected_center_x, mesh.vertices[:, 1] - used.protected_center_y)
    inside = radius < used.protected_radius - 0.2
    assert np.any(inside)
    assert np.all(mesh.vertices[inside, 2] <= used.min_height + 0.05)
    high = mesh.vertices[mesh.vertices[:, 2] > np.quantile(mesh.vertices[:, 2], 0.92)]
    mean_az = design_xy_to_sound_azimuth(high[:, 0].mean(), high[:, 1].mean(), 0.0)
    assert angular_error_deg(mean_az, 0.0, 0.0, 0.0) < 20.0


def test_rotation_moves_the_ridge():
    field = blob_field(0.0, 0.0, spread_deg=12.0)
    mesh, _response, used, _warnings = generate_architecture(field, _params(mode="terrain", rotation_offset_deg=90.0))
    high = mesh.vertices[mesh.vertices[:, 2] > np.quantile(mesh.vertices[:, 2], 0.92)]
    assert high[:, 0].mean() < -2.0


def test_height_and_intensity_change_the_mesh():
    field = blob_field(0.0)
    short, *_rest = generate_architecture(field, _params(mode="terrain", max_height=3.0))
    tall, *_rest = generate_architecture(field, _params(mode="terrain", max_height=12.0))
    quiet, *_rest = generate_architecture(field, _params(mode="terrain", response_intensity=0.0))
    assert tall.vertices[:, 2].max() > short.vertices[:, 2].max() + 4.0
    assert quiet.vertices[:, 2].max() == pytest.approx(0.0, abs=0.05)


def test_rectangle_site_stays_inside_its_boundary():
    field = blob_field(20.0)
    mesh, _response, used, _warnings = generate_architecture(field, _params(mode="terrain", site_shape="rectangle"))
    assert np.all(np.abs(mesh.vertices[:, 0]) <= used.site_width * 0.5 + 1e-6)
    assert np.all(np.abs(mesh.vertices[:, 1]) <= used.site_depth * 0.5 + 1e-6)


def test_walls_are_taller_toward_the_loud_direction_and_leave_the_zone_clear():
    field = blob_field(0.0, spread_deg=10.0)
    mesh, _response, used, warnings = generate_architecture(field, _params(mode="walls"))
    assert any("heuristic" in note.lower() or "line-of-sight" in note.lower() for note in warnings)
    front = _sector_height(mesh, (used.protected_center_x, used.protected_center_y), 0.0)
    back = _sector_height(mesh, (used.protected_center_x, used.protected_center_y), 180.0)
    assert front > back + 1.0
    radius = np.hypot(mesh.vertices[:, 0] - used.protected_center_x, mesh.vertices[:, 1] - used.protected_center_y)
    inside = radius < used.protected_radius - 0.15
    assert np.all(mesh.vertices[inside, 2] <= 0.05)


def test_pavilion_encloses_the_loud_side_and_keeps_headroom():
    field = blob_field(0.0, spread_deg=10.0)
    mesh, _response, used, _warnings = generate_architecture(field, _params(mode="pavilion", max_height=6.0))
    center = np.array([used.protected_center_x, used.protected_center_y])
    radius = np.hypot(mesh.vertices[:, 0] - center[0], mesh.vertices[:, 1] - center[1])
    inside = radius < used.protected_radius * 0.65
    low_obstruction = inside & (mesh.vertices[:, 2] > 0.2) & (mesh.vertices[:, 2] < 2.0)
    assert not np.any(low_obstruction)
    assert np.any(inside & (mesh.vertices[:, 2] > 2.2))
    outer = used.pavilion_diameter * 0.5
    shell = (radius > outer - used.wall_thickness - 0.3) & (radius < outer + 0.3)
    front_az = design_xy_to_sound_azimuth(
        mesh.vertices[shell, 0] - center[0],
        mesh.vertices[shell, 1] - center[1],
        0.0,
    )
    front = mesh.vertices[shell][np.abs(((front_az + 180) % 360) - 180) < 25]
    back = mesh.vertices[shell][np.abs(((front_az - 180 + 180) % 360) - 180) < 25]
    assert front[:, 2].max() > back[:, 2].max() + 0.4
    assert _component_count(mesh) == 1


def _component_count(mesh) -> int:
    """Connected vertex components via shared faces, without a graph library."""
    parent = np.arange(len(mesh.vertices))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for face in mesh.faces:
        union(int(face[0]), int(face[1]))
        union(int(face[0]), int(face[2]))
    roots = {find(int(index)) for face in mesh.faces for index in face}
    return len(roots)


def test_protection_heuristic_blocks_the_loud_direction_only():
    field = blob_field(0.0, spread_deg=8.0)
    mesh, response, used, _warnings = generate_architecture(field, _params(mode="walls", openness=0.55, entry_width_deg=40))
    report = estimate_protection(mesh, response, used, {"broadband": field.aggregate("broadband")})
    assert "not a sound pressure reduction" in report.disclaimer
    horizontal = np.argmin(np.abs(response.elevation_deg))
    front = int(np.argmin(np.abs(response.azimuth_deg - 2.5)))
    back = int(np.argmin(np.abs(((response.azimuth_deg - 180.0 + 180) % 360) - 180)))
    assert report.coverage[horizontal, front] == 1.0
    assert report.coverage[horizontal, back] == 0.0
    assert report.exposure_after[horizontal, front] < report.exposure_before[horizontal, front]
    assert any("Heuristic" in warning or "unobstructed" in warning or "heuristic" in warning.lower() for warning in report.warnings)
