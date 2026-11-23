from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist
from PySide6.QtWidgets import QApplication
from soundform.geometry.point_filters import merge_close_points, subtract_curve_tubes
from soundform.geometry.site_model import SiteModel
from soundform.ui.main_window import MainWindow


def test_merge_keeps_surface_representatives_without_chain_collapse():
    p = pv.PolyData(np.array([[0.,0,0],[.05,0,0],[.1,0,0],[1,0,0]]))
    p["energy"] = np.arange(4.)
    p["Normals"] = np.tile([0.,0,1], (4,1))
    result = merge_close_points(p, .06)
    assert np.allclose(result.points[:,0], [0,.1,1])
    assert np.array_equal(result["energy"], [0,2,3])
    assert pdist(result.points).min() > .06
    assert merge_close_points(p, 0).n_points == 4
    empty = merge_close_points(pv.PolyData(), .1)
    assert empty.n_points == 0


def test_curve_subtraction_is_3d_preserves_attributes_and_handles_empty():
    p = pv.PolyData(np.array([[.5,0,0],[.5,.2,0],[.5,0,.5],[2,0,0]], float))
    p["energy"] = np.arange(4.)
    vertices = np.array([[0,0,0],[1,0,0]], float)
    curve = (vertices, np.array([2,0,1]), .25)
    result, count = subtract_curve_tubes(p, [curve])
    assert count == 2
    assert np.array_equal(result["energy"], [2,3])
    assert np.array_equal(result.points, p.points[[2,3]])
    assert subtract_curve_tubes(p, [(vertices, np.array([2,0,1]), 0)])[1] == 0
    assert subtract_curve_tubes(p, [(vertices, np.array([2,0,1]), 10)])[0].n_points == 0


def test_cloud_tables_subtraction_origin_move_and_layout():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        route = dict(name="Walk", category="Circulation", layer="Circulation",
            vertices=np.array([[0.,0,0],[1,0,0]]), faces=np.empty((0,3), int), lines=np.array([2,0,1]))
        site = SiteModel(Path("test.3dm"), np.zeros(3), {}, [], [],
            origins=[("A", np.zeros(3)), ("B", np.array([3.,0,0]))], objects=[route])
        window._set_site(site)
        assert window.cloud_origins.rowCount() == 2
        assert window.circulation_table.rowCount() == 1
        assert window.circulation_table.cellWidget(0,1).value() == 2
        window.cloud_origins.cellWidget(1,1).setChecked(False)
        window.cloud_origins.cellWidget(0,2).setValue(1.5)
        window._create_cloud()
        assert window.viewport._cloud_members == (0,)
        assert window.viewport._origin_settings[0]["radius_scale"] == 1.5
        window.viewport._cloud_points = pv.PolyData(np.array([[.5,0,0],[.5,0,3.]]))
        window.viewport._cloud_points["energy"] = np.array([.1,.9])
        window.viewport._reconstructions["delaunay"] = pv.Sphere()
        window._subtract_circulation_points()
        assert window.viewport._cloud_points.n_points == 1
        assert not window.viewport._reconstructions
        sampled = pv.Plane(i_resolution=4, j_resolution=4)
        sampled["energy"] = np.ones(sampled.n_points)
        window.viewport._cloud_sampled = sampled
        count, capped = window.viewport.generate_cloud_points(100, 100, 0, .2)
        assert 1 < count < 100
        assert pdist(window.viewport._cloud_points.points).min() > .2
        assert "Normals" in window.viewport._cloud_points.point_data
        window._select_viewport_object(dict(name="A", category="Pavilion Origin", point=np.zeros(3), origin_index=0))
        for box, value in zip(window.origin_xyz, [1.,2.,3.]):
            box.setValue(value)
        window._edit_origin_position()
        assert np.allclose(site.origins[0][1], [1,2,3])
        assert window.viewport._cloud_members == (0,)
        assert window.viewport._origin_settings[0]["radius_scale"] == 1.5
        assert window.viewport._cloud_points is None
        assert "Generate" in window.hex_layout._panels
        generate = window.hex_layout._panels["Generate"][0]
        assert generate.isAncestorOf(window.delaunay_max_edge)
        cloud = window.hex_layout._panels["Cloud"][0]
        assert cloud.isAncestorOf(window.cloud_merge_distance)
        window.hex_layout.resize(1200, 800)
        window.hex_layout._arrange()
    finally:
        window.close()
