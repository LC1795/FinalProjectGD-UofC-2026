import numpy as np
import pyvista as pv
import rhino3dm as rhino
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from soundform.ui.main_window import MainWindow
from soundform.geometry.site_model import SiteModel
from soundform.export.selection_export import export_selection
from soundform.errors import ExportError
from pathlib import Path


def test_selected_mixed_geometry_exports_to_rhino_and_obj(tmp_path):
    mesh = pv.Plane().triangulate()
    mesh.cell_data["center_energy"] = np.full(mesh.n_cells,.7)
    line = pv.PolyData(np.array([[1.,0,0],[2,0,0],[2,1,0]]))
    line.verts = np.empty(0,int)
    line.lines = np.array([3,0,1,2])
    points = pv.PolyData(np.array([[3.,0,0]]))
    items = [("Tile",mesh),("Route",line),("Origin",points)]
    export_selection(items,tmp_path/"selected.3dm")
    loaded = rhino.File3dm.Read(str(tmp_path/"selected.3dm"))
    assert loaded.Settings.ModelUnitSystem == rhino.UnitSystem.Meters
    assert len(loaded.Objects) == 3
    assert [o.Attributes.Name for o in loaded.Objects] == ["Tile","Route","Origin"]
    assert len(loaded.Objects[0].Geometry.VertexColors) > 0
    export_selection(items,tmp_path/"selected.obj")
    text = (tmp_path/"selected.obj").read_text()
    assert "o Route" in text and "l " in text and "p " in text
    with pytest.raises(ExportError,match="STL supports"):
        export_selection(items,tmp_path/"selected.stl")
    cloud = pv.PolyData(np.array([[0.,0,0],[1,2,3.]]))
    export_selection([("Samples",cloud)],tmp_path/"points.3dm")
    assert len(rhino.File3dm.Read(str(tmp_path/"points.3dm")).Objects[0].Geometry) == 2
    export_selection([items[0]],tmp_path/"selected.stl")
    assert (tmp_path/"selected.stl").stat().st_size > 80


def test_focus_uses_selection_never_hover_and_default_sphere_is_three_metres():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        site = SiteModel(Path("site.3dm"),np.zeros(3),{},[],[],origins=[("A",np.array([10.,0,0]))])
        w._set_site(site)
        def no_pick(*args,**kwargs):
            raise AssertionError("Framing must not pick the hovered object")
        w.viewport._pick_object = no_pick
        before = w.viewport.plotter.camera_position
        w.viewport.focus_object(QPointF(20,20))
        assert w.viewport.plotter.camera_position == before
        w._select_viewport_object(dict(category="Pavilion Origin",name="A",point=site.origins[0][1],origin_index=0))
        assert w.origin_radius.value() == 3
        w.viewport.focus_object(QPointF(200,200))
        assert np.allclose(w.viewport.plotter.camera.focal_point,[10,0,0])
        datasets = w._selection_datasets()
        assert len(datasets)==1 and datasets[0][1].n_points==1
        # The actual analysis sphere uses the same radius even when pavilion size changes.
        params = w.design_state.parameters()
        params.pavilion_diameter = 20
        w.viewport.set_overlay(None,None,None,params)
        sphere = w.viewport.plotter.actors["analysis-fill-0"].mapper.dataset
        assert np.ptp(sphere.points[:,0]) == pytest.approx(6,rel=.003)
        w._select_viewport_object(None)
        before = w.viewport.plotter.camera_position
        w.viewport.focus_object(QPointF(20,20))
        assert w.viewport.plotter.camera_position == before
    finally:
        w.close()
