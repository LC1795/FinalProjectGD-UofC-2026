"""Resolve selected scene geometry for export."""
import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox
from soundform.geometry.pavilion_edit import selected_mesh
from soundform.geometry.analysis_envelope import merged_envelope, clip_to_terrain
from soundform.export.selection_export import export_selection


class SelectionActionsMixin:
    def _selection_datasets(self, items=None):
        if items is None:
            active = self._selected_object
            if active is None:
                return []
            items = []
            if active.get("from_layers"):
                for node in self.layers_tree.selectedItems():
                    items.extend(self._layer_entries(node))
            if not items:
                items = [active]
        result, seen = [], set()
        for item in items:
            if not item:
                continue
            category = item["category"]
            key = (category, item.get("name"), item.get("origin_index"), item.get("_actor"), item.get("_scene_id"))
            if key in seen:
                continue
            seen.add(key)
            geometry = None
            if category == "Pavilion" and self.mesh is not None:
                label = item.get("name")
                if label is None and "cell" in item:
                    from soundform.geometry.pavilion_edit import part_labels
                    label = part_labels(self.mesh)[item["cell"]]
                part = selected_mesh(self.mesh, label)
                if part is not None:
                    geometry = pv.PolyData(part.vertices, np.c_[np.full(len(part.faces),3), part.faces].ravel())
            elif "_scene_id" in item:
                geometry = self._scene_objects[item["_scene_id"]]["mesh"]
            elif category == "Analysis envelope":
                geometry = self.viewport._cloud_surface
            elif category == "Surface points":
                geometry = self.viewport._cloud_points
            elif category == "Reconstruction":
                geometry = self.viewport._reconstructions.get(item["_method"])
            elif category == "Analysis sphere" and self.site_model:
                index = item["origin_index"]
                actor = self.viewport.plotter.actors.get(f"analysis-fill-{index}")
                if actor is not None:
                    geometry = actor.mapper.dataset
                else:
                    point = self.site_model.origins[index][1]+np.array([0,0,1.6])
                    radius = self.viewport._origin_settings.get(index, {}).get("radius",3.0)
                    if index in (self.viewport._cloud_members or ()):
                        radius *= self.viewport._origin_settings.get(index, {}).get("radius_scale",1)
                    geometry = clip_to_terrain(merged_envelope([point],radius,merge=False,full_sphere=True),self.viewport._terrain)
            elif "vertices" in item:
                faces = np.c_[np.full(len(item["faces"]),3),item["faces"]].ravel()
                geometry = pv.PolyData(item["vertices"],faces)
                if "lines" in item:
                    geometry.verts = np.empty(0,dtype=np.int64)
                    geometry.lines = item["lines"]
                if item.get("curve_type")=="spline":
                    geometry.field_data["curve_type"]=np.array(["spline"])
                    geometry.field_data["control_points"]=item["control_points"]
            elif "point" in item:
                geometry = pv.PolyData(np.asarray([item["point"]]))
            if geometry is not None and geometry.n_points:
                result.append((item.get("name",category),geometry.copy()))
        return result

    def _export_selected(self):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before exporting.")
            return
        datasets = self._selection_datasets()
        if not datasets:
            QMessageBox.information(self,"Nothing selected","Select an object in the viewport or Layers first.")
            return
        path, chosen = QFileDialog.getSaveFileName(self,"Export selected","soundform-selected.3dm",
            "Rhino model (*.3dm);;Wavefront OBJ (*.obj);;STL mesh (*.stl)")
        if not path:
            return
        from pathlib import Path
        if not Path(path).suffix:
            path += ".obj" if "*.obj" in chosen else ".stl" if "*.stl" in chosen else ".3dm"
        try:
            info = export_selection(datasets,path)
        except Exception as error:
            QMessageBox.warning(self,"Export failed",str(error))
            return
        self.statusBar().showMessage(f"Exported {info['objects']} selected object(s) to {info['path']}.")
