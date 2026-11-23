"""Commit viewport sketches to the chosen scene layer."""
from uuid import uuid4
import numpy as np
from PySide6.QtWidgets import QMessageBox
from soundform.geometry.drawing import drawing_geometry


class DrawingActionsMixin:
    def _start_drawing(self,kind):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before drawing.")
            self.hex_layout._sync_draw_buttons(self.viewport._drawing_mode or "")
            return
        if kind:
            self.viewport.begin_drawing(kind)
        else:
            self.viewport.cancel_drawing()

    def _refresh_drawing_layers(self):
        if not hasattr(self,"hex_layout"):
            return
        layers={self.layers_tree.topLevelItem(i).text(0) for i in range(self.layers_tree.topLevelItemCount())}
        layers.add("Drawing")
        if self.site_model:
            layers.update(("Circulation","Pavilion Origin","Sound Sources"))
        self.hex_layout.set_drawing_layers(sorted(layers))

    def _drawing_name(self,prefix):
        names={o["name"] for o in self._scene_objects.values()}
        if self.site_model:
            names.update(o["name"] for o in self.site_model.objects)
            names.update(name for name,_ in self.site_model.origins+self.site_model.sources)
        index=1
        while f"{prefix} {index}" in names:
            index+=1
        return f"{prefix} {index}"

    def _commit_drawing(self,kind,points):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Drawing was not added because another task started. Finish that task and draw again.")
            return
        layer=self.hex_layout.drawing_layer.currentText() or "Drawing"
        pieces=[np.asarray([p]) for p in points] if kind=="point" else [points]
        latest=None
        try:
            for controls in pieces:
                mesh=drawing_geometry(kind,controls)
                name=self._drawing_name(kind.title())
                category=next((part for part in layer.split("::") if part in ("Circulation","Pavilion Origin","Sound Sources")),None)
                if self.site_model and kind=="point" and category in ("Pavilion Origin","Sound Sources"):
                    collection=self.site_model.origins if category=="Pavilion Origin" else self.site_model.sources
                    collection.append((name,np.asarray(controls[0]).copy()))
                    self._refresh_edited_site()
                    index=len(collection)-1
                    latest=dict(name=name,category=category,point=controls[0],
                                **({"origin_index":index} if category=="Pavilion Origin" else {"_source_index":index,"_actor":f"source-{index}"}))
                elif self.site_model and category=="Circulation" and kind!="point":
                    obj=dict(name=name,layer=layer,category="Circulation",vertices=mesh.points.copy(),
                             faces=np.empty((0,3),int),lines=mesh.lines.copy(),circulation_radius=2,
                             curve_type=kind,control_points=np.asarray(controls).copy())
                    self.site_model.objects.append(obj)
                    index=len(self.site_model.objects)-1
                    actor=f"context-object-{index}"
                    self.viewport._replace(actor,mesh,color="#f2bc65",line_width=3,lighting=False)
                    self.viewport._pick_objects[actor]=obj
                    self._sync_cloud_tables()
                    latest={**obj,"_site_index":index,"_actor":actor}
                else:
                    identifier=uuid4().hex
                    self._scene_objects[identifier]=dict(name=name,layer=layer,mesh=mesh)
                    self._show_scene_object(identifier)
                    latest=self.viewport._pick_objects["scene-"+identifier]
        except Exception as error:
            QMessageBox.warning(self,"Could not add drawing",str(error))
            return
        self._empty_layers.add(layer)
        self._rebuild_layers()
        if latest:
            self._select_viewport_object(latest)
        self.viewport._apply_visibility()
        self.viewport.plotter.render()
        self.statusBar().showMessage(f"Added {len(pieces)} {kind} object(s) to {layer}.")
