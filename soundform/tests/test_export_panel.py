from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolButton
from soundform.ui.main_window import MainWindow
from soundform.geometry.site_model import SiteModel
from soundform.export.selection_export import export_selection


def test_export_layer_checkboxes_and_layout(tmp_path):
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        obj=dict(name="Walk",layer="Circulation",category="Circulation",
            vertices=np.array([[0.,0,0],[1,0,0]]),faces=np.empty((0,3),int),lines=np.array([2,0,1]))
        w._set_site(SiteModel(Path("test.3dm"),np.zeros(3),{},[],[],origins=[("A",np.zeros(3))],objects=[obj]))
        assert "Manual" not in w.hex_layout._panels and "Export" in w.hex_layout._panels
        assert not hasattr(w,"connection_max_distance")
        assert "Connect points" not in [t.text() for t in w.hex_layout.findChildren(QToolButton)]
        root=next(w.export_tree.topLevelItem(i) for i in range(w.export_tree.topLevelItemCount()) if w.export_tree.topLevelItem(i).text(0)=="Circulation")
        root.setCheckState(0,Qt.Checked)
        datasets=w._checked_export_datasets()
        assert len(datasets)==1 and datasets[0][1].n_lines==1
        export_selection(datasets,tmp_path/"walk.3dm")
        w._rebuild_layers()
        assert len(w._checked_export_datasets())==1
        w._check_exports(False)
        assert not w._checked_export_datasets()
        assert w.export_format.count()==3
        w.show();app.processEvents()
        expected={"Audio":5,"Analysis":5,"Layers":4,"Properties":4,"Export":2}
        for name,edge in expected.items():
            assert w.hex_layout._panels[name][1]._edge_anchor[0]==edge
        assert len(w.hex_layout._outer_vertices)==6
        assert w.hex_layout._band_width>0
    finally:
        w.close()
