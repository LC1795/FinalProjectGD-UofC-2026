"""Regression coverage for layer clipboard edits and dependent origin indices."""
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from soundform.geometry.site_model import SiteModel
from soundform.geometry.pavilion_edit import part_labels
from soundform.ui.main_window import MainWindow


def test_layer_clipboard_survives_deletion_and_remaps_origins():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        model = SiteModel(Path("test.3dm"), np.zeros(3), {}, [], [], origins=[
            ("A", np.zeros(3)), ("B", np.array([5., 0, 0]))])
        window._set_site(model)
        def root(name):
            tree = window.layers_tree
            return next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                        if tree.topLevelItem(i).text(0) == name)
        window.viewport._origin_settings = {0: {"radius": 3}, 1: {"radius": 7}}
        window._copy_layer_entries(root("Pavilion Origin").child(0))
        window._delete_layer_entries(root("Pavilion Origin").child(0))
        assert [n for n, _ in model.origins] == ["B"]
        assert window.viewport._origin_settings == {0: {"radius": 7}}
        window._paste_layer_entries(root("Pavilion Origin"))
        assert model.origins[1][0] == "A copy"
        assert window.viewport._origin_settings[1]["radius"] == 3
        # Copy an entire layer, then delete and paste into its retained empty row.
        window._copy_layer_entries(root("Pavilion Origin"))
        window._delete_layer_entries(root("Pavilion Origin"))
        assert root("Pavilion Origin").childCount() == 0
        assert not any(name.startswith("analysis-fill-") for name in window.viewport.plotter.actors)
        window._paste_layer_entries(root("Pavilion Origin"))
        assert len(model.origins) == 2
        assert not np.shares_memory(window._layer_clipboard[0]["point"], model.origins[0][1])
        mesh = trimesh.creation.box()
        mesh.face_attributes["pavilion_part"] = np.full(len(mesh.faces), "Wall")
        window.mesh = mesh
        window._show_editable_mesh(mesh)
        window._copy_layer_entries(root("Pavilion").child(0))
        window._delete_layer_entries(root("Pavilion"))
        assert window.mesh is None
        window._paste_layer_entries(root("Pavilion"))
        assert set(part_labels(window.mesh)) == {"Wall copy"}
        assert len(window.mesh.faces) == 12
        assert not root("Pavilion").isExpanded()
    finally:
        window.close()
        app.processEvents()
