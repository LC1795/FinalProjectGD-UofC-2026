import numpy as np
import pyvista as pv
from soundform.geometry.voronoi_filters import subtract_circulation
from soundform.geometry.cloud_reconstruction import reconstruct_cloud


def test_pipe_cuts_triangle_interior_and_keeps_tile_metadata():
    plane = pv.Plane(i_size=4, j_size=4, i_resolution=1, j_resolution=1).triangulate()
    plane.cell_data["center_energy"] = np.full(plane.n_cells,.7)
    plane.cell_data["voronoi_cell"] = np.full(plane.n_cells,3)
    curve = (np.array([[.4,0,-1],[.4,0,1.]]), np.array([2,0,1]), .3)
    clipped = subtract_circulation(plane, [curve])
    assert 0 < clipped.area < plane.area
    _, nearest = clipped.find_closest_cell([.4,0,0], return_closest_point=True)
    assert np.linalg.norm(nearest-np.array([.4,0,0])) >= .3-1e-8
    assert np.allclose(clipped.cell_data["center_energy"], .7)
    assert np.all(clipped.cell_data["voronoi_cell"] == 3)
    assert subtract_circulation(plane, [(curve[0],curve[1],0)]).area == plane.area


def test_preserve_openings_toggle_and_complete_removal():
    sphere = pv.Sphere(theta_resolution=12,phi_resolution=10)
    points = pv.PolyData(sphere.points[::3])
    points["energy"] = np.linspace(0,1,points.n_points)
    curve = (np.array([[0.,0,-2],[0,0,2]]),np.array([2,0,1]),.2)
    common = dict(boundary=sphere,voronoi_gap=0,voronoi_cells=0,circulation_curves=[curve])
    original = reconstruct_cloud(points, "voronoi", preserve_circulation=False, **common)
    cut = reconstruct_cloud(points, "voronoi", preserve_circulation=True, **common)
    assert cut.area < original.area
    assert np.allclose(cut.cell_data["center_energy"],points["energy"][cut.cell_data["center_source_point"]])
    import pytest
    with pytest.raises(ValueError,match="removed every"):
        reconstruct_cloud(points,"voronoi",boundary=sphere,circulation_curves=[(curve[0],curve[1],3)])


def test_straight_subdivision_does_not_multiply_geometry():
    plane = pv.Plane(i_size=4,j_size=4,i_resolution=8,j_resolution=8).triangulate()
    plane.cell_data["center_energy"] = np.linspace(0,1,plane.n_cells)
    plane.cell_data["voronoi_cell"] = np.arange(plane.n_cells)
    results = []
    for count in (1,100):
        route = np.c_[np.linspace(-2,2,count+1),np.zeros(count+1),np.zeros(count+1)]
        results.append(subtract_circulation(plane,[(route,np.r_[count+1,np.arange(count+1)],.3)]))
    assert results[0].n_cells == results[1].n_cells
    assert abs(results[0].area-results[1].area)<1e-9
    for result in results:
        assert np.allclose(result.cell_data["center_energy"],plane.cell_data["center_energy"][result.cell_data["voronoi_cell"]])


def test_bent_and_overlapping_paths_keep_clearance_and_remote_triangles():
    plane = pv.Plane(i_size=6,j_size=6,i_resolution=8,j_resolution=8).triangulate()
    plane.cell_data["voronoi_cell"] = np.arange(plane.n_cells)
    first = np.array([[-1.,0,0],[0,0,0],[0,1,0]])
    second = np.array([[-1.,.2,0],[1,.2,0]])
    curves = [(first,np.array([3,0,1,2]),.25),(second,np.array([2,0,1]),.25)]
    cut = subtract_circulation(plane,curves)
    for a,b in [(first[0],first[1]),(first[1],first[2]),(second[0],second[1])]:
        for fraction in np.linspace(0,1,9):
            p=a+(b-a)*fraction
            _,closest=cut.find_closest_cell(p,return_closest_point=True)
            assert np.linalg.norm(closest-p)>=.25-1e-8
    remote = np.array([2.8,2.8,0])
    _,closest=cut.find_closest_cell(remote,return_closest_point=True)
    assert np.linalg.norm(closest-remote)<1e-8
    # Avoid needless rebuilding when every pipe is far away.
    far=subtract_circulation(plane,[(second+100,np.array([2,0,1]),.25)])
    assert np.array_equal(far.points,plane.points)
    assert np.array_equal(far.faces,plane.faces)
