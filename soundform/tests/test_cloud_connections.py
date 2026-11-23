import itertools

import numpy as np
import pyvista as pv
import pytest

from soundform.geometry.cloud_connections import connect_cloud_points


def test_connections_enforce_distances_and_preserve_point_energy():
    points = pv.PolyData(np.array(list(itertools.product((0., 1.), repeat=3))))
    points["energy"] = np.linspace(0, 1, 8)
    mesh = connect_cloud_points(points, max_distance=1.01, max_connections=64)
    assert mesh.n_lines == 12
    assert mesh.n_cells == mesh.n_lines
    assert np.allclose(mesh.cell_data["length_m"], 1)
    assert np.allclose(mesh["energy"], points["energy"][mesh["source_point"]])
    diagonal = connect_cloud_points(points, min_distance=1.1, max_distance=1.5, max_connections=64)
    assert diagonal.n_lines == 12
    assert np.allclose(diagonal.cell_data["length_m"], np.sqrt(2))


def test_connection_limits_and_no_duplicate_edges():
    points = pv.PolyData(np.random.default_rng(1).random((80, 3)))
    mesh = connect_cloud_points(points, max_distance=2, max_connections=3, max_lines=20)
    edges = mesh.lines.reshape(-1, 3)[:, 1:]
    assert mesh.n_lines == 20
    assert mesh.field_data["line_limit_reached"][0]
    assert np.bincount(edges.ravel()).max() <= 3
    assert len(np.unique(np.sort(edges, axis=1), axis=0)) == 20
    assert np.all(edges[:, 0] != edges[:, 1])


def test_two_points_and_empty_or_invalid_distance_ranges():
    points = pv.PolyData(np.array([[0.,0,0], [1.,0,0]]))
    assert connect_cloud_points(points, max_distance=1).n_lines == 1
    with pytest.raises(ValueError, match="No point pairs"):
        connect_cloud_points(points, max_distance=.5)
    with pytest.raises(ValueError, match="Minimum distance"):
        connect_cloud_points(points, min_distance=2, max_distance=1)


def test_crossing_check_distinguishes_shared_ends_overlaps_and_projection():
    from soundform.geometry.connection_checks import CrossingIndex
    xyz = np.array([[0.,0,0],[2.,0,0],[1.,-1,0],[1.,1,0],[0.,1,0],[1.,0,0]])
    index = CrossingIndex(xyz, 1, .001)
    index.add(xyz[[0,1]], (0,1))
    assert index.crosses(xyz[[2,3]], (2,3))
    assert not index.crosses(xyz[[0,4]], (0,4))
    assert index.crosses(xyz[[0,5]], (0,5))
    assert not index.crosses(xyz[[2,3]] + [0,0,.1], (2,3))


def test_adaptive_spacing_rejects_bridge_out_of_dense_cluster():
    points = pv.PolyData(np.c_[[0.,.1,.2,.3,2.],np.zeros(5),np.zeros(5)])
    mesh = connect_cloud_points(points,max_distance=3,adaptive=True,spacing_factor=1.5)
    assert mesh['length_m'].max() <= .45
    assert 4 not in mesh['source_point']
    assert mesh.field_data['rejected_spacing'][0] > 0


def test_surface_routing_never_links_close_disconnected_sheets():
    lower = pv.Plane(i_size=2,j_size=2,i_resolution=2,j_resolution=2)
    upper = lower.copy()
    upper.points[:,2] += .05
    surface = lower.merge(upper,merge_points=False)
    xy = np.array([[-.5,-.5],[.5,-.5],[-.5,.5],[.5,.5]])
    points = pv.PolyData(np.vstack((np.c_[xy,np.zeros(4)],np.c_[xy,np.full(4,.05)])))
    mesh = connect_cloud_points(points,max_distance=2,surface=surface,surface_only=True)
    first = mesh['start_point']
    second = mesh['end_point']
    assert np.all((first<4)==(second<4))
    assert mesh.n_lines > 0
    distance = mesh.compute_implicit_distance(surface)['implicit_distance']
    assert np.abs(distance).max() < 1e-6
    assert mesh['length_m'].max() <= 2
