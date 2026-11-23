import numpy as np
import pyvista as pv
import pytest
from soundform.geometry.cloud_reconstruction import reconstruct_cloud, _voronoi


def test_reach_clips_triangle_interiors_and_leaves_sparse_surface_empty():
    surface = pv.Plane(i_size=4, j_size=4, i_resolution=1, j_resolution=1)
    sites = np.array([[0.,0,0]])
    unlimited = _voronoi(sites, surface, cell_count=0, gap=0)
    limited = _voronoi(sites, surface, cell_count=0, gap=0, max_reach=.5)
    assert unlimited.area == pytest.approx(16)
    assert limited.area == pytest.approx(.5, rel=.004)
    assert np.linalg.norm(limited.points, axis=1).max() <= .5+1e-9
    assert limited.area < unlimited.area*.1
    with pytest.raises(ValueError, match="nonnegative"):
        _voronoi(sites, surface, max_reach=-1)


def test_center_intensity_survives_subsampling_duplicates_and_reach():
    surface = pv.Sphere(theta_resolution=16, phi_resolution=12)
    xyz = np.vstack([surface.points[::5], surface.points[0]])
    points = pv.PolyData(xyz)
    points["energy"] = np.linspace(0, 1, len(xyz))
    result = reconstruct_cloud(points, "voronoi", boundary=surface,
        voronoi_cells=12, voronoi_seed=17, voronoi_gap=.01, voronoi_max_reach=.2)
    source = result.cell_data["center_source_point"]
    assert np.unique(source).size <= 12
    assert len(xyz)-1 not in source
    assert np.array_equal(result.cell_data["center_energy"], points["energy"][source])
    triangles = result.points[result.faces.reshape(-1, 4)[:,1:]]
    assert np.linalg.norm(triangles-xyz[source,None,:], axis=2).max() <= .2+1e-8
    for cell in np.unique(result["voronoi_cell"]):
        assert len(np.unique(result.cell_data["center_energy"][result["voronoi_cell"]==cell])) == 1
    points["energy"][:] = .8
    other = reconstruct_cloud(points, "voronoi", boundary=surface,
        voronoi_cells=12, voronoi_seed=17, voronoi_gap=.01, voronoi_max_reach=.2)
    assert np.array_equal(result.points, other.points)
    assert np.array_equal(result.faces, other.faces)
    assert np.allclose(other.cell_data["center_energy"], .8)


def test_reach_closure_is_straight_between_whole_tile_edge_intersections():
    surface = pv.Plane(center=(.75,0,0), i_size=2.5, j_size=1, i_resolution=6, j_resolution=4)
    m = _voronoi(np.array([[0.,0,0]]), surface, cell_count=0, gap=0, max_reach=1)
    cut = np.sqrt(1-.5**2)
    assert m.points[:,0].max() == pytest.approx(cut, abs=1e-7)
    assert m.area == pytest.approx(cut+.5, abs=1e-6)
    assert np.linalg.norm(m.points,axis=1).max() <= 1+1e-8
