import numpy as np
import pyvista as pv
import pytest
from soundform.geometry.cloud_reconstruction import reconstruct_cloud


def samples(surface):
    p = pv.PolyData(surface.points.copy())
    p["energy"] = np.linspace(0, 1, p.n_points)
    return p


def test_sphere_wraps_and_ignores_sound_data():
    surface = pv.Sphere(theta_resolution=18, phi_resolution=14)
    p = samples(surface)
    a = reconstruct_cloud(p, "delaunay", boundary=surface)
    p["energy"][:] = 10000
    b = reconstruct_cloud(p, "delaunay", boundary=surface)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.faces, b.faces)
    assert "energy" not in a.point_data
    assert a.n_cells > 100
    assert a.points[:, 2].min() < -.45 and a.points[:, 2].max() > .45
    assert np.array_equal(a.points, p.points[a["source_point"]])
    edges = a.extract_all_edges()
    assert edges.n_cells > a.n_points
    assert np.linalg.norm(a.points, axis=1).min() > .49


def test_disconnected_nearby_sheets_do_not_connect():
    lower = pv.Plane(i_resolution=5, j_resolution=5)
    upper = lower.copy()
    upper.points[:, 2] += .01
    surface = lower.merge(upper, merge_points=False)
    m = reconstruct_cloud(samples(surface), "delaunay", boundary=surface)
    triangles = m.points[m.faces.reshape(-1, 4)[:, 1:]]
    assert np.ptp(triangles[:, :, 2], axis=1).max() < 1e-8
    assert np.unique(m.points[:, 2]).size == 2


def test_surface_cap_and_missing_surface():
    surface = pv.Sphere(theta_resolution=18, phi_resolution=14)
    p = samples(surface)
    m = reconstruct_cloud(p, "delaunay", boundary=surface, max_edge=.25)
    assert m["max_surface_edge_m"].max() <= .25
    t = m.points[m.faces.reshape(-1, 4)[:, 1:]]
    assert np.linalg.norm(t-np.roll(t, 1, axis=1), axis=2).max() <= .25+1e-8
    with pytest.raises(ValueError, match="No surface triangles"):
        reconstruct_cloud(p, "delaunay", boundary=surface, max_edge=.001)
