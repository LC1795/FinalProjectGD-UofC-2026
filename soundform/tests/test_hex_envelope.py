"""Regression checks for merged envelopes and hexagonal panel boundaries."""

import numpy as np
from PySide6.QtCore import QPointF

from soundform.geometry.analysis_envelope import merged_envelope, envelope_energy
from soundform.ui.hex_layout import hex_vertices, sector_polygon, content_rectangle


def _polygon_area(polygon):
    points = np.array([(point.x(), point.y()) for point in polygon])
    return abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                      - points[:, 1] * np.roll(points[:, 0], -1))) / 2


def test_hex_sectors_cover_screen_outside_viewport_without_overlapping():
    center = QPointF(720, 450)
    vertices = hex_vertices(center, 300)
    sectors = [sector_polygon(center, vertices[i], vertices[(i + 1) % 6], 1440, 900)
               for i in range(6)]
    assert np.isclose(sum(map(_polygon_area, sectors)) + _polygon_area(vertices), 1440 * 900)
    for polygon in sectors:
        x, y, width, height = content_rectangle(polygon)
        assert width > 0 and height > 0
        for dx in (1, width - 1):
            for dy in (1, height - 1):
                from PySide6.QtCore import Qt
                assert polygon.containsPoint(QPointF(x + dx, y + dy), Qt.OddEvenFill)


def test_nearby_envelopes_merge_while_distant_ones_stay_separate():
    near = merged_envelope([[0, 0, 1.6], [9, 0, 1.6]], 6)
    distant = merged_envelope([[0, 0, 1.6], [40, 0, 1.6]], 6)
    assert near.connectivity().point_data["RegionId"].max() == 0
    assert distant.connectivity().point_data["RegionId"].max() == 1
    assert near.points[:, 2].min() >= 1.6 - 1e-5


def test_one_origin_retains_radius_and_silence_is_preserved():
    surface = merged_envelope([[0, 0, 1.6]], 6)
    assert np.isclose(surface.points[:, 0].max(), 6, atol=0.15)
    values = envelope_energy(surface.points, [[0, 0, 1.6]],
                             np.arange(0, 360, 30), np.array([0, 45, 90]),
                             np.zeros((3, 12)), 0)
    assert np.all(values == 0)


def test_surface_energy_is_continuous_across_origin_boundary():
    points = np.array([[4.999, 3, 3], [5.001, 3, 3]])
    azimuth = np.arange(0, 360, 30)
    scores = np.tile((1 + np.cos(np.deg2rad(azimuth))) / 2, (3, 1))
    energy = envelope_energy(points, [[0, 0, 1.6], [10, 0, 1.6]],
                             azimuth, np.array([0, 45, 90]), scores, 0)
    assert abs(energy[0] - energy[1]) < 0.001


def test_surface_analysis_uses_triangle_areas_and_threshold():
    from soundform.geometry.analysis_envelope import analyze_envelope
    surface = merged_envelope([[0, 0, 1.6]], 6)
    surface.point_data["energy"] = np.full(surface.n_points, 0.5)
    result = analyze_envelope(surface, -12)
    assert result["area_m2"] > 0
    assert result["mean_relative_energy"] == 0.5
    assert result["above_threshold_area_percent"] == 100
    assert 1 <= len(result["samples"]) <= 49

def test_connection_strength_reaches_farther_without_changing_single_sphere_radius():
    weak = merged_envelope([[0, 0, 1.6], [18, 0, 1.6]], 6, strength=0.1)
    strong = merged_envelope([[0, 0, 1.6], [18, 0, 1.6]], 6, strength=3)
    assert weak.connectivity().point_data["RegionId"].max() == 1
    assert strong.connectivity().point_data["RegionId"].max() == 0
    for strength in (0.1, 3):
        surface = merged_envelope([[0, 0, 1.6]], 6, strength=strength)
        assert np.isclose(surface.points[:, 0].max(), 6, atol=0.2)

def test_envelope_base_follows_sloping_terrain():
    import pyvista as pv
    from soundform.geometry.analysis_envelope import clip_to_terrain
    sphere = merged_envelope([[0, 0, 1.6]], 6, full_sphere=True)
    terrain = pv.Plane(i_size=40, j_size=40, i_resolution=8, j_resolution=8)
    terrain.points[:, 2] = 0.2 * terrain.points[:, 0]
    clipped = clip_to_terrain(sphere, terrain)
    clearance = clipped.points[:, 2] - 0.2 * clipped.points[:, 0]
    assert clipped.n_points > 0
    assert clearance.min() >= -1e-5
    assert np.any(np.abs(clearance) < 1e-4)
    assert clipped.points[:, 2].min() < 1.6


def test_sound_point_density_and_surface_placement():
    from soundform.geometry.analysis_envelope import clip_to_terrain, scatter_surface_points
    sphere = clip_to_terrain(merged_envelope([[0, 0, 1.6]], 6, full_sphere=True))
    sphere.point_data["energy"] = (sphere.points[:, 0] > 0).astype(float)
    points, capped = scatter_surface_points(sphere, 0.1, 4)
    assert not capped and points.n_points > 100
    assert np.sum(points.points[:, 0] > 0) > 8 * np.sum(points.points[:, 0] < 0)
    assert points.points[:, 2].min() >= -1e-5
    distance = points.compute_implicit_distance(sphere).point_data["implicit_distance"]
    assert np.abs(distance).max() < 1e-4


def test_point_threshold_clips_triangles_before_sampling():
    import pyvista as pv
    from soundform.geometry.analysis_envelope import scatter_surface_points
    surface = pv.Sphere(radius=3, theta_resolution=40, phi_resolution=30)
    surface.point_data["energy"] = (surface.points[:, 0] + 3) / 6
    points, _ = scatter_surface_points(surface, 2, 8, energy_threshold=0.7)
    assert points.n_points > 50
    assert points.point_data["energy"].min() >= 0.7 - 1e-6
    assert points.points[:, 0].min() >= 1.2 - 1e-5
    assert np.allclose(np.linalg.norm(points.point_data["Normals"], axis=1), 1)
    surface.point_data["energy"][:] = 0.5
    empty, _ = scatter_surface_points(surface, 2, 8, energy_threshold=0.8)
    assert empty.n_points == 0


def test_poisson_reconstruct_oriented_sphere():
    import pyvista as pv
    from soundform.geometry.analysis_envelope import scatter_surface_points
    from soundform.geometry.cloud_reconstruction import reconstruct_cloud
    surface = pv.Sphere(radius=3, theta_resolution=40, phi_resolution=30)
    surface.point_data["energy"] = np.full(surface.n_points, 0.6)
    points, _ = scatter_surface_points(surface, 15, 15)
    for method in ("poisson",):
        mesh = reconstruct_cloud(points, method, resolution=48)
        assert mesh.n_cells > 100
        assert np.isfinite(mesh.points).all()
        radii = np.linalg.norm(mesh.points, axis=1)
        assert np.abs(radii - 3).mean() < 0.3
        assert np.max(radii) < 3.6
        assert np.allclose(mesh.point_data["energy"], 0.6)


def test_voronoi_cells_follow_cloud_surface():
    import pyvista as pv
    from soundform.geometry.analysis_envelope import scatter_surface_points
    from soundform.geometry.cloud_reconstruction import reconstruct_cloud
    sphere = pv.Sphere(radius=3, theta_resolution=24, phi_resolution=20)
    sphere.point_data["energy"] = np.full(sphere.n_points, 0.5)
    points, _ = scatter_surface_points(sphere, 0.5, 0.5)
    mesh = reconstruct_cloud(points, "voronoi", boundary=sphere)
    assert mesh.n_cells > 100
    assert np.isfinite(mesh.points).all()
    assert np.linalg.norm(mesh.points, axis=1).max() <= 3.01
    assert np.linalg.norm(mesh.points, axis=1).min() > 2.9
    distance = mesh.compute_implicit_distance(sphere)["implicit_distance"]
    assert np.abs(distance).max() < 1e-5
    assert len(np.unique(mesh.cell_data["voronoi_cell"])) > 10
    assert np.allclose(mesh.point_data["energy"], 0.5)


def test_surface_voronoi_gap_and_seed_controls():
    import pyvista as pv
    from soundform.geometry.analysis_envelope import scatter_surface_points
    from soundform.geometry.cloud_reconstruction import reconstruct_cloud
    surface = pv.Sphere(radius=3, theta_resolution=20, phi_resolution=16)
    surface["energy"] = np.full(surface.n_points, 0.5)
    points, _ = scatter_surface_points(surface, 1, 1)
    tiled = reconstruct_cloud(points, "voronoi", boundary=surface, voronoi_cells=20, voronoi_gap=0)
    gapped = reconstruct_cloud(points, "voronoi", boundary=surface, voronoi_cells=20, voronoi_gap=0.15)
    repeat = reconstruct_cloud(points, "voronoi", boundary=surface, voronoi_cells=20, voronoi_gap=0.15)
    changed = reconstruct_cloud(points, "voronoi", boundary=surface, voronoi_cells=20, voronoi_gap=0.15, voronoi_seed=9)
    assert np.isclose(tiled.area, surface.area, rtol=1e-5)
    assert 0 < gapped.area < tiled.area * .95
    assert len(np.unique(tiled.cell_data["voronoi_cell"])) == 20
    assert np.allclose(gapped.points, repeat.points)
    assert changed.n_points != gapped.n_points or not np.allclose(changed.points, gapped.points)
