import numpy as np
import pyvista as pv
import pytest
from scipy.spatial import ConvexHull
from soundform.geometry.cloud_reconstruction import reconstruct_cloud


def cloud(xyz):
    p = pv.PolyData(xyz)
    p["energy"] = np.arange(len(xyz), dtype=float)
    return p


def test_global_triangulation_tiles_hull_without_overlap():
    uv = np.random.default_rng(42).random((100, 2))
    xyz = np.c_[uv, np.sin(uv[:, 0])]
    xyz = np.vstack([xyz, xyz[0], xyz[0] + [0, 0, 10]])
    p = cloud(xyz)
    m = reconstruct_cloud(p, "delaunay")
    t = m.points[m.faces.reshape(-1, 4)[:, 1:]][:, :, :2]
    area = np.abs((t[:, 1, 0]-t[:, 0, 0])*(t[:, 2, 1]-t[:, 0, 1]) -
                  (t[:, 1, 1]-t[:, 0, 1])*(t[:, 2, 0]-t[:, 0, 0])) / 2
    assert area.sum() == pytest.approx(ConvexHull(uv).volume)
    # Explicitly check every pair: triangle interiors must not intersect.
    for i, a in enumerate(t):
        for b in t[i+1:]:
            separated = False
            for tri in (a, b):
                for edge in np.roll(tri, -1, axis=0)-tri:
                    axis = np.array([-edge[1], edge[0]])
                    pa, pb = a @ axis, b @ axis
                    if pa.max() <= pb.min()+1e-12 or pb.max() <= pa.min()+1e-12:
                        separated = True
            assert separated
    assert m.n_points == 100
    assert np.array_equal(m.points, p.points[m["source_point"]])
    assert "energy" not in m.point_data


def test_edge_cutoff_removes_whole_triangles_and_leaves_distant_point_unused():
    x, y = np.meshgrid(np.arange(4.), np.arange(4.))
    xyz = np.vstack([np.c_[x.ravel(), y.ravel(), np.zeros(16)], [100, 100, 0]])
    m = reconstruct_cloud(cloud(xyz), "delaunay", max_edge=1.5)
    t = m.points[m.faces.reshape(-1, 4)[:, 1:]]
    assert np.linalg.norm(t-np.roll(t, 1, axis=1), axis=2).max() <= 1.5
    assert 16 not in m["source_point"]
    assert m.n_cells == 18
    with pytest.raises(ValueError, match="No triangles"):
        reconstruct_cloud(cloud(xyz), "delaunay", max_edge=.5)


def test_projection_choices_and_collinear_error():
    x, z = np.meshgrid(np.arange(4.), np.arange(4.))
    p = cloud(np.c_[x.ravel(), np.zeros(16), z.ravel()])
    with pytest.raises(ValueError, match="collinear"):
        reconstruct_cloud(p, "delaunay", projection="xy")
    for projection in ("xz", "best_fit"):
        assert reconstruct_cloud(p, "delaunay", projection=projection).n_cells == 18
