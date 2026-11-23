import numpy as np
from PySide6.QtCore import Qt,QRect,QPoint
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from soundform.ui.main_window import MainWindow


def test_box_selection_and_keyboard_delete():
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        w.show();app.processEvents()
        w._commit_drawing("point",np.array([[0.,0,0],[1,0,0]]))
        view=w.viewport.plotter.interactor
        w.viewport.plotter.render_window.SetSize(view.width(),view.height())
        w.viewport.plotter.reset_camera()
        app.processEvents()
        h=w.hex_layout
        assert h.drawing_layer.y()<min(b.y() for b in h._draw_buttons.values())
        # Exercise the actual drag gesture, selection synchronization, and Delete.
        QTest.mousePress(view,Qt.LeftButton,pos=QPoint(1,1))
        QTest.mouseMove(view,QPoint(view.width()-2,view.height()-2))
        QTest.mouseRelease(view,Qt.LeftButton,pos=QPoint(view.width()-2,view.height()-2))
        assert len(w._selection_datasets())==2
        assert len([n for n in w.viewport.plotter.actors if n.startswith("box-selection-")])==2
        QTest.keyClick(view,Qt.Key_Delete)
        assert not w._scene_objects
        w._commit_drawing("point",np.array([[0.,0,0]]))
        QTest.keyClick(view,Qt.Key_Backspace)
        assert not w._scene_objects
        QTest.keyClick(view,Qt.Key_Backspace)
    finally:
        w.close()
        app.processEvents()


def test_box_selection_add_and_clear():
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        w.show();app.processEvents()
        w._commit_drawing("point",np.array([[0.,0,0],[1,0,0]]))
        view=w.viewport.plotter.interactor
        w.viewport.plotter.render_window.SetSize(view.width(),view.height())
        w.viewport.plotter.reset_camera()
        app.processEvents()
        w._box_select(QRect(0,0,view.width(),view.height()))
        assert len(w._selection_datasets())==2
        w._box_select(QRect(-100,-100,1,1),True)
        assert len(w._selection_datasets())==2
        w._box_select(QRect(-100,-100,1,1))
        assert not w._selection_datasets()
        assert not any(n.startswith("box-selection-") for n in w.viewport.plotter.actors)
    finally:
        w.close()
        app.processEvents()


def test_keyboard_delete_removes_imported_curve():
    from pathlib import Path
    from soundform.geometry.site_model import SiteModel
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        curve=dict(name="Route",layer="Circulation",category="Circulation",
                   vertices=np.array([[0.,0,0],[1,0,0]]),faces=np.empty((0,3),int),
                   lines=np.array([2,0,1]))
        w._set_site(SiteModel(Path("site.3dm"),np.zeros(3),{},[],[],objects=[curve]))
        w._select_viewport_object(w.viewport._pick_objects["context-object-0"])
        QTest.keyClick(w.viewport.plotter.interactor,Qt.Key_Delete)
        assert not w.site_model.objects
    finally:
        w.close()
        app.processEvents()
