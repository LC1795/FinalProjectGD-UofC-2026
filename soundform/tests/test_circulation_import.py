import numpy as np
import rhino3dm as rhino
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from soundform.geometry.site_model import load_site
from soundform.ui.main_window import MainWindow


def test_curves_import_and_remain_lines_through_clipboard(tmp_path):
    file = rhino.File3dm()
    file.Settings.ModelUnitSystem = rhino.UnitSystem.Millimeters
    def layer(name):
        item = rhino.Layer()
        item.Name = name
        return file.Layers.Add(item)
    origin_layer = layer("Pavilion Origin")
    circulation = layer("Circulation")
    attr = rhino.ObjectAttributes()
    attr.LayerIndex = origin_layer
    file.Objects.AddPoint(rhino.Point3d(1000, 2000, 0), attr)
    attr = rhino.ObjectAttributes()
    attr.LayerIndex = circulation
    attr.Name = "Route"
    file.Objects.AddCurve(rhino.PolylineCurve([rhino.Point3d(1000,2000,0), rhino.Point3d(2000,2000,0), rhino.Point3d(2000,3000,0)]), attr)
    attr.Name = "Loop"
    file.Objects.AddCurve(rhino.Circle(rhino.Point3d(1000,2000,0), 500).ToNurbsCurve(), attr)
    path = tmp_path / "curves.3dm"
    assert file.Write(str(path), 8)
    model = load_site(path)
    assert len(model.objects) == 2
    assert np.allclose(model.objects[0]["vertices"], [[0,0,0],[1,0,0],[1,1,0]])
    loop = model.objects[1]["vertices"]
    assert np.allclose(loop[0], loop[-1])
    assert np.max(np.abs(np.linalg.norm(loop[:,:2],axis=1)-.5)) < .001
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._set_site(model)
        actor = window.viewport.plotter.actors["context-object-0"]
        assert actor.mapper.dataset.n_lines == 1
        tree = window.layers_tree
        root = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0)=="Circulation")
        window._copy_layer_entries(root.child(0))
        window._paste_layer_entries(root)
        assert len(model.objects) == 3
        assert "lines" in model.objects[-1]
        assert window.viewport.plotter.actors["context-object-2"].mapper.dataset.n_lines == 1
    finally:
        window.close()


def test_main_window_closes_all_open_notes():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    window.hex_layout.set_collapsed(False)
    app.processEvents()
    panels = [panel for panel, _ in window.hex_layout._panels.values()]
    assert all(panel.isVisible() for panel in panels)
    assert window.close()
    app.processEvents()
    assert all(not panel.isVisible() for panel in panels)
    assert not window.isVisible()
