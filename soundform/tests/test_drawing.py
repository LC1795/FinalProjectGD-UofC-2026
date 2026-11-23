from pathlib import Path
import numpy as np
import rhino3dm as rhino
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from soundform.geometry.drawing import drawing_geometry,intersect_plane
from soundform.geometry.site_model import SiteModel
from soundform.ui.main_window import MainWindow
from soundform.export.selection_export import export_selection


def test_spline_has_control_points_and_exports_as_true_nurbs(tmp_path):
    controls=np.array([[0.,0,0],[1,2,0],[2,2,0],[3,0,0]])
    spline=drawing_geometry("spline",controls)
    assert np.allclose(spline.points[[0,-1]],controls[[0,-1]])
    assert np.array_equal(spline.field_data["control_points"],controls)
    export_selection([("Spline",spline)],tmp_path/"spline.3dm")
    geom=rhino.File3dm.Read(str(tmp_path/"spline.3dm")).Objects[0].Geometry
    assert isinstance(geom,rhino.NurbsCurve)
    assert geom.Degree==3
    assert not geom.IsPolyline()
    assert np.allclose(intersect_plane(np.array([1.,2,3]),np.array([1.,2,-3]),np.zeros(3),np.array([0.,0,1])),[1,2,0])


def test_click_finish_cancel_and_layer_assignment(monkeypatch):
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        w.show();app.processEvents()
        h=w.hex_layout
        assert h._panels["Generate"][1]._edge_anchor[0]==0
        assert h.drawing_layer.currentText()=="Drawing"
        view=w.viewport.plotter.interactor
        monkeypatch.setattr(w.viewport,"_drawing_pick",lambda pos:np.array([pos.x()/100,pos.y()/100,0.]))
        h._draw_buttons["line"].click()
        QTest.mouseClick(view,Qt.LeftButton,pos=QPoint(120,120))
        QTest.mouseClick(view,Qt.LeftButton,pos=QPoint(180,150))
        assert len(w.viewport._drawing_points)==2
        h.drawing_layer.setFocus()
        QTest.keyClick(h.drawing_layer,Qt.Key_Return)
        assert w.viewport._drawing_mode is None
        assert len(w._scene_objects)==1
        line=next(iter(w._scene_objects.values()))
        assert line["layer"]=="Drawing" and line["mesh"].n_lines==1
        h._draw_buttons["spline"].click()
        for point in (QPoint(100,100),QPoint(150,140),QPoint(210,110)):
            QTest.mouseClick(view,Qt.LeftButton,pos=point)
        QTest.mouseClick(view,Qt.RightButton,pos=QPoint(210,110))
        assert len(w._scene_objects)==2
        h._draw_buttons["point"].click()
        QTest.mouseClick(view,Qt.LeftButton,pos=QPoint(130,130))
        QTest.keyClick(view,Qt.Key_Escape)
        assert len(w._scene_objects)==2
        assert w.viewport._drawing_mode is None
        assert "drawing-preview" not in w.viewport.plotter.actors
        assert not any(b.isChecked() for b in h._draw_buttons.values())
        h._draw_buttons["point"].click()
        QTest.mouseClick(view,Qt.LeftButton,pos=QPoint(130,130))
        QTest.mouseClick(view,Qt.LeftButton,pos=QPoint(160,160))
        QTest.mouseClick(view,Qt.RightButton,pos=QPoint(160,160))
        assert len(w._scene_objects)==4
        assert w._selected_object["category"]=="Scene object"
    finally:
        w.close()


def test_drawn_circulation_spline_and_origin_are_functional(tmp_path):
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        w._set_site(SiteModel(Path("site.3dm"),np.zeros(3),{},[],[],origins=[("A",np.zeros(3))]))
        w.hex_layout.drawing_layer.setCurrentText("Circulation")
        controls=np.array([[0.,0,0],[1,1,0],[2,0,0]])
        w._commit_drawing("spline",controls)
        obj=w.site_model.objects[-1]
        assert obj["category"]=="Circulation" and obj["curve_type"]=="spline"
        assert w.circulation_table.rowCount()==1
        assert w.circulation_table.cellWidget(0,1).value()==2
        result=w._selection_datasets()
        export_selection(result,tmp_path/"route.3dm")
        geom=rhino.File3dm.Read(str(tmp_path/"route.3dm")).Objects[0].Geometry
        assert isinstance(geom,rhino.NurbsCurve)
        w.hex_layout.drawing_layer.setCurrentText("Pavilion Origin")
        w._commit_drawing("point",np.array([[3.,4,2]]))
        assert len(w.site_model.origins)==2
        assert np.allclose(w.site_model.origins[-1][1],[3,4,2])
        assert w.cloud_origins.rowCount()==2
    finally:
        w.close()


def test_view_plane_fallback_is_finite_for_all_camera_views():
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        w.show();app.processEvents()
        interactor=w.viewport.plotter.interactor
        for name,axis in (("top",2),("front",1),("right",0),("perspective",2)):
            w.viewport.set_view(name)
            result=w.viewport._drawing_pick(QPointF(interactor.width()/2,interactor.height()/2))
            assert result is not None and np.isfinite(result).all()
            expected=w.viewport.plotter.camera.focal_point[axis] if name in ("front","right") else 0
            assert abs(result[axis]-expected)<1e-7
    finally:
        w.close()
