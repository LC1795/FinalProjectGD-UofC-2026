"""Layer tree clipboard and scene editing commands."""
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from soundform.geometry.pavilion_edit import edit_mesh, selected_mesh, part_labels


class LayerActionsMixin:
    def _layer_context_menu(self, position):
        item = self.layers_tree.itemAt(position)
        if item is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before editing the scene.")
            return
        menu = QMenu(self.layers_tree)
        objects = self._layer_entries(item)
        copy_action = menu.addAction("Copy")
        copy_action.setEnabled(bool(objects))
        delete_action = menu.addAction("Delete")
        delete_action.setEnabled(bool(objects))
        menu.addSeparator()
        paste_action = menu.addAction("Paste into this layer")
        paste_action.setEnabled(bool(self._layer_clipboard))
        chosen = menu.exec(self.layers_tree.viewport().mapToGlobal(position))
        if chosen == copy_action:
            self._copy_layer_entries(item)
        elif chosen == delete_action:
            self._delete_layer_entries(item)
        elif chosen == paste_action:
            self._paste_layer_entries(item)

    def _layer_entries(self, item):
        if item.parent() is None:
            return [item.child(i).data(0, Qt.UserRole) for i in range(item.childCount())]
        return [item.data(0, Qt.UserRole)]

    def _copy_layer_entries(self, item):
        snapshots = []
        for data in self._layer_entries(item):
            category = data["category"]
            snapshot = dict(name=data["name"], category=category, layer=data.get("layer", item.text(0)))
            if category == "Pavilion":
                snapshot["pavilion"] = selected_mesh(self.mesh, data["name"]).copy()
            elif category == "Pavilion Origin":
                snapshot["point"] = np.array(data["point"], copy=True)
                snapshot["settings"] = deepcopy(self.viewport._origin_settings.get(data["origin_index"], {}))
            elif "_source_index" in data:
                snapshot["point"] = np.array(data["point"], copy=True)
            else:
                mesh = None
                if "_scene_id" in data:
                    mesh = self._scene_objects[data["_scene_id"]]["mesh"]
                elif category == "Analysis envelope":
                    mesh = self.viewport._cloud_surface
                elif category == "Surface points":
                    mesh = self.viewport._cloud_points
                elif category == "Reconstruction":
                    mesh = self.viewport._reconstructions.get(data["_method"])
                elif category == "Analysis sphere":
                    actor = self.viewport.plotter.actors.get("analysis-fill-" + str(data["origin_index"]))
                    if actor is None:
                        from soundform.geometry.analysis_envelope import merged_envelope, clip_to_terrain
                        index = data["origin_index"]
                        point = self.site_model.origins[index][1] + np.array([0, 0, 1.6])
                        radius = self.viewport._origin_settings.get(index, {}).get("radius", 3.0)
                        mesh = clip_to_terrain(merged_envelope([point], radius, merge=False, full_sphere=True), self.viewport._terrain)
                    else:
                        mesh = actor.mapper.dataset
                elif "vertices" in data:
                    faces = np.c_[np.full(len(data["faces"]), 3), data["faces"]].ravel()
                    mesh = pv.PolyData(data["vertices"], faces)
                    if "lines" in data:
                        mesh.verts = np.empty(0, dtype=np.int64)
                        mesh.lines = data["lines"]
                    if data.get("curve_type")=="spline":
                        mesh.field_data["curve_type"]=np.array(["spline"])
                        mesh.field_data["control_points"]=data["control_points"]
                if mesh is not None:
                    snapshot["mesh"] = mesh.copy(deep=True)
                else:
                    continue
            snapshots.append(snapshot)
        self._layer_clipboard = snapshots
        self.statusBar().showMessage(f"Copied {len(snapshots)} object(s). Right-click a layer to paste.")

    def _paste_layer_entries(self, item):
        root = item if item.parent() is None else item.parent()
        layer = root.text(0)
        site_changed = False
        pavilion_changed = False
        for source in self._layer_clipboard:
            obj = deepcopy(source)
            existing = {root.child(i).text(0) for i in range(root.childCount())}
            existing.update(v["name"] for v in self._scene_objects.values() if v["layer"] == layer)
            if self.site_model:
                existing.update(o["name"] for o in self.site_model.objects)
                existing.update(name for name, _ in self.site_model.origins + self.site_model.sources)
            if self.mesh is not None:
                existing.update(part_labels(self.mesh))
            name = obj["name"] + " copy"
            number = 2
            while name in existing:
                name = obj["name"] + f" copy {number}"
                number += 1
            if "pavilion" in obj and layer == "Pavilion":
                import trimesh
                mesh = obj["pavilion"]
                mesh.face_attributes["pavilion_part"] = np.full(len(mesh.faces), name)
                if not pavilion_changed:
                    self._mesh_undo.append(self.mesh)
                if self.mesh is None:
                    self.mesh = mesh
                else:
                    labels = np.r_[part_labels(self.mesh), part_labels(mesh)]
                    self.mesh = trimesh.util.concatenate([self.mesh, mesh])
                    self.mesh.face_attributes["pavilion_part"] = labels
                pavilion_changed = True
            elif "point" in obj and self.site_model and layer in ("Pavilion Origin", "Sound Sources"):
                collection = self.site_model.origins if layer == "Pavilion Origin" else self.site_model.sources
                collection.append((name, obj["point"].copy()))
                if layer == "Pavilion Origin":
                    self._pending_origin_settings[len(collection)-1] = obj.get("settings", {})
                site_changed = True
            elif obj.get("category") in ("Terrain", "Solids", "Vegetation", "Circulation") and "mesh" in obj and self.site_model:
                mesh = obj["mesh"]
                if len(mesh.faces):
                    mesh = mesh.triangulate(pass_lines=True, pass_verts=True)
                category = layer.split("::")[0]
                if category not in ("Terrain", "Solids", "Vegetation", "Circulation"):
                    category = obj["category"]
                entry = dict(name=name, layer=layer, category=category,
                    vertices=mesh.points.copy(), faces=mesh.faces.reshape(-1, 4)[:, 1:].copy())
                if mesh.n_lines:
                    entry["lines"] = mesh.lines.copy()
                if "curve_type" in mesh.field_data and str(mesh.field_data["curve_type"][0])=="spline":
                    entry["curve_type"]="spline"
                    entry["control_points"]=np.asarray(mesh.field_data["control_points"]).copy()
                self.site_model.objects.append(entry)
                site_changed = True
            else:
                if "pavilion" in obj:
                    mesh = obj["pavilion"]
                    obj["mesh"] = pv.PolyData(mesh.vertices, np.c_[np.full(len(mesh.faces),3),mesh.faces].ravel())
                elif "point" in obj:
                    obj["mesh"] = pv.PolyData(np.array([obj["point"]]))
                identifier = uuid4().hex
                self._scene_objects[identifier] = dict(name=name, layer=layer, mesh=obj["mesh"])
                self._show_scene_object(identifier)
        if site_changed:
            self._refresh_edited_site()
        if pavilion_changed:
            self._pavilion_changed()
        self._empty_layers.add(layer)
        self._rebuild_layers()
        self.statusBar().showMessage(f"Pasted {len(self._layer_clipboard)} object(s) into {layer}. Copies overlap their originals.")

    def _delete_layer_entries(self, item):
        entries = self._layer_entries(item)
        root = item if item.parent() is None else item.parent()
        layer = root.text(0)
        self._empty_layers.add(layer)
        self._delete_entries(entries,layer)

    def _delete_entries(self,entries,layer="selection"):
        self._empty_layers.update(data.get("layer",data["category"]) for data in entries)
        site_indices, origin_indices, source_indices = set(), set(), set()
        site_changed = False
        pavilion_changed = False
        for data in entries:
            if "_scene_id" in data:
                identifier = data["_scene_id"]
                self._scene_objects.pop(identifier, None)
                self.viewport._pick_objects.pop("scene-" + identifier, None)
                self.viewport._drop("scene-" + identifier)
            elif "_site_index" in data:
                site_indices.add(data["_site_index"])
            elif data["category"] == "Pavilion Origin":
                origin_indices.add(data["origin_index"])
            elif "_source_index" in data:
                source_indices.add(data["_source_index"])
            elif data["category"] == "Pavilion":
                if not pavilion_changed:
                    self._mesh_undo.append(self.mesh)
                self.mesh = edit_mesh(self.mesh, data["name"], delete=True)
                pavilion_changed = True
            elif data["category"] == "Analysis sphere":
                self.viewport._deleted_spheres.add(data["origin_index"])
                self.viewport._envelope_key = None
            elif data["category"] == "Analysis envelope":
                self.viewport._cloud_members = None
                self.viewport._metaball_enabled = False
                self.viewport._envelope_key = None
            elif data["category"] == "Surface points":
                self.viewport._cloud_points = None
                self.viewport._drop("cloud-points")
                self.viewport.clear_reconstructions()
            elif data["category"] == "Reconstruction":
                method = data["_method"]
                self.viewport.remove_reconstruction(method)
        if self.site_model and (site_indices or origin_indices or source_indices):
            model = self.site_model
            model.objects = [o for i,o in enumerate(model.objects) if i not in site_indices]
            old = self.viewport._origin_settings
            keep = [i for i in range(len(model.origins)) if i not in origin_indices]
            self._pending_origin_settings = {j:deepcopy(old.get(i,{})) for j,i in enumerate(keep)}
            self.viewport._deleted_spheres = {j for j,i in enumerate(keep) if i in self.viewport._deleted_spheres}
            model.origins = [model.origins[i] for i in keep]
            model.sources = [o for i,o in enumerate(model.sources) if i not in source_indices]
            self.viewport._origin_settings = deepcopy(self._pending_origin_settings)
            self._refresh_edited_site()
            site_changed = True
        self._selected_origins.clear()
        self._select_viewport_object(None)
        if pavilion_changed:
            self._pavilion_changed()
        if not site_changed:
            self._refresh_overlay(self.design_state.parameters())
        self.metaball_box.blockSignals(True)
        self.metaball_box.setChecked(self.viewport._metaball_enabled)
        self.metaball_box.setEnabled(bool(self.viewport._cloud_members))
        self.metaball_box.blockSignals(False)
        self._rebuild_layers()
        self.statusBar().showMessage(f"Deleted {len(entries)} object(s) from {layer}.")

    def _refresh_edited_site(self):
        model = self.site_model
        model.meshes = {}
        for category in ("Terrain", "Solids", "Vegetation", "Circulation"):
            pieces = [o for o in model.objects if o["category"] == category and len(o["faces"])]
            if not pieces:
                continue
            vertices, faces, offset = [], [], 0
            for obj in pieces:
                vertices.append(obj["vertices"])
                faces.append(obj["faces"] + offset)
                offset += len(obj["vertices"])
            model.meshes[category] = (np.vstack(vertices), np.vstack(faces))
        settings = deepcopy(self.viewport._origin_settings)
        settings.update(self._pending_origin_settings)
        self._pending_origin_settings.clear()
        self._set_site(model)
        self.viewport._origin_settings = settings
        self._sync_cloud_tables()
        for identifier in self._scene_objects:
            self._show_scene_object(identifier)
        self._refresh_overlay(self.design_state.parameters())

    def _show_scene_object(self, identifier):
        obj = self._scene_objects[identifier]
        mesh = obj["mesh"]
        data = dict(name=obj["name"], layer=obj["layer"], category="Scene object",
                    _scene_id=identifier, _actor="scene-" + identifier,
                    vertices=mesh.points, faces=mesh.faces.reshape(-1,4)[:,1:] if len(mesh.faces) else np.empty((0,3),int))
        options = dict(show_scalar_bar=False)
        if mesh.n_faces:
            options.update(color="#d9c7a6", opacity=.9)
        elif mesh.n_lines:
            options.update(line_width=2, color="#f2bc65", lighting=False)
        else:
            options.update(style="points", render_points_as_spheres=True, point_size=6, color="#f2bc65")
        self.viewport._replace("scene-" + identifier, mesh, **options)
        self.viewport._pick_objects["scene-" + identifier] = data
        self.viewport._apply_visibility()
        self.viewport.plotter.render()
