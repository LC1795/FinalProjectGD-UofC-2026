from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QItemSelectionModel
from PySide6.QtWidgets import QApplication
from soundform.geometry.site_model import SiteModel
from soundform.ui.main_window import MainWindow


def test_layer_navigation_survives_origin_movement_and_scene_refresh():
    app=QApplication.instance() or QApplication([])
    w=MainWindow()
    try:
        objects=[dict(name=f"Route {i}",layer="Circulation",category="Circulation",
            vertices=np.array([[float(i),0,0],[float(i),1,0]]),faces=np.empty((0,3),int),lines=np.array([2,0,1])) for i in range(35)]
        w._set_site(SiteModel(Path("test.3dm"),np.zeros(3),{},[],[],origins=[("A",np.zeros(3)),("B",np.array([4.,0,0]))],objects=objects))
        w.show();w.hex_layout.set_open("Layers",True);app.processEvents()
        tree=w.layers_tree
        def root(name):
            return next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0)==name)
        root("Circulation").setExpanded(True)
        root("Analysis spheres").setExpanded(True)
        root("Pavilion Origin").setExpanded(False)
        tree.setCurrentItem(root("Analysis spheres").child(0))
        root("Circulation").child(1).setSelected(True)
        root("Circulation").child(2).setCheckState(0,Qt.Unchecked)
        sphere = root("Analysis spheres").child(0)
        tree.setCurrentItem(sphere,0,QItemSelectionModel.NoUpdate)
        w._select_viewport_object({**sphere.data(0,Qt.UserRole),"from_layers":True})
        app.processEvents()
        tree.verticalScrollBar().setValue(12)
        app.processEvents()
        scroll=tree.verticalScrollBar().value()
        assert scroll>0
        selected={item.text(0) for item in tree.selectedItems()}
        current=tree.currentItem().text(0)
        w.origin_xyz[2].setValue(2.5)
        w._edit_origin_position()
        app.processEvents()
        assert root("Circulation").isExpanded()
        assert root("Analysis spheres").isExpanded()
        assert not root("Pavilion Origin").isExpanded()
        assert tree.verticalScrollBar().value()==scroll
        assert {item.text(0) for item in tree.selectedItems()}==selected
        assert tree.currentItem().text(0)==current
        assert root("Circulation").child(2).checkState(0)==Qt.Unchecked
        assert w._selected_object["category"]=="Analysis sphere"
        assert w.site_model.origins[0][1][2]==2.5
        # New scene rows must not collapse existing groups or change selection.
        w._empty_layers.add("Extra layer")
        w._rebuild_layers();app.processEvents()
        assert root("Circulation").isExpanded()
        assert tree.verticalScrollBar().value()==scroll
        assert {item.text(0) for item in tree.selectedItems()}==selected
        assert tree.currentItem().text(0)==current
    finally:
        w.close()
