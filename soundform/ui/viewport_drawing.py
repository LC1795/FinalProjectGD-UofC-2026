"""Viewport sketch gestures and preview; no scene changes until Finish."""
import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt,QEvent
from soundform.geometry.drawing import drawing_geometry,intersect_plane


class ViewportDrawingMixin:
    def begin_drawing(self,kind):
        self.cancel_drawing()
        if kind not in ("point","line","spline"):
            return
        self._drawing_mode=kind
        self._drawing_points=[]
        self._drag=None
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.setFocus(Qt.OtherFocusReason)
        self.plotter.interactor.setCursor(Qt.CrossCursor)
        self.drawing_mode_changed.emit(kind)
        self.drawing_hint.emit("Left-click to place "+("points" if kind=="point" else "control points")+"; right-click or Enter to finish; Escape to cancel. Picks visible surfaces, otherwise the current view's drawing plane.")

    def cancel_drawing(self):
        self._drawing_mode=None
        self._drawing_points=[]
        self._drop("drawing-preview")
        self._drop("drawing-control-points")
        self.plotter.interactor.unsetCursor()
        self.drawing_mode_changed.emit("")
        self.plotter.render()

    def finish_drawing(self):
        kind=self._drawing_mode
        if not kind:
            return
        minimum={"point":1,"line":2,"spline":3}[kind]
        if len(self._drawing_points)<minimum:
            self.drawing_hint.emit(f"Add at least {minimum} points to finish this {kind}, or press Escape to cancel.")
            return
        points=np.asarray(self._drawing_points).copy()
        self.cancel_drawing()
        self.geometry_created.emit(kind,points)

    def _drawing_event(self,event):
        if not self._drawing_mode:
            return False
        if event.type()==QEvent.KeyPress:
            if event.key()==Qt.Key_Escape:
                self.cancel_drawing()
                self.drawing_hint.emit("Drawing cancelled.")
                return True
            if event.key() in (Qt.Key_Return,Qt.Key_Enter):
                self.finish_drawing()
                return True
        if event.type() in (QEvent.MouseButtonPress,QEvent.MouseButtonDblClick,QEvent.MouseButtonRelease):
            if event.button()==Qt.RightButton:
                if event.type()==QEvent.MouseButtonPress:
                    self.finish_drawing()
                return True
            if event.button()==Qt.LeftButton:
                if event.type()==QEvent.MouseButtonPress:
                    point=self._drawing_pick(event.position())
                    if point is not None and (not self._drawing_points or np.linalg.norm(point-self._drawing_points[-1])>1e-9):
                        self._drawing_points.append(point)
                        self._drawing_preview()
                return True
        if event.type()==QEvent.MouseMove:
            if self._drawing_points:
                self._drawing_preview(self._drawing_pick(event.position()))
            return True
        return False

    def _drawing_preview(self,hover=None):
        points=np.asarray(self._drawing_points)
        if not len(points):
            return
        self._replace("drawing-control-points",pv.PolyData(points),style="points",
            point_size=9,render_points_as_spheres=True,color="#f2bc65",lighting=False)
        if self._drawing_mode!="point":
            controls=np.vstack([points,hover]) if hover is not None else points
            kind="spline" if self._drawing_mode=="spline" and len(controls)>=3 else "line"
            if len(controls)>=2:
                self._replace("drawing-preview",drawing_geometry(kind,controls),line_width=3,color="#c6cc9d",lighting=False)
        self.plotter.render()

    def _drawing_pick(self,position):
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        ratio=self.plotter.interactor.devicePixelRatioF()
        x=position.x()*ratio
        y=(self.plotter.interactor.height()-position.y())*ratio-1
        picker=vtkCellPicker()
        picker.PickFromListOn()
        picker.SetTolerance(.005)
        count=0
        for name,actor in self.plotter.actors.items():
            if not actor.GetVisibility() or not name.startswith(("context-object-","context-Terrain","scene-","reconstruction-","sound-field")):
                continue
            data=actor.mapper.dataset
            if hasattr(data,"faces") and len(data.faces):
                picker.AddPickList(actor)
                count+=1
        if count and picker.Pick(x,y,0,self.plotter.renderer):
            return np.asarray(picker.GetPickPosition(),dtype=float)
        renderer=self.plotter.renderer
        world=[]
        for z in (0.,1.):
            renderer.SetDisplayPoint(x,y,z)
            renderer.DisplayToWorld()
            value=np.asarray(renderer.GetWorldPoint())
            if abs(value[3])<1e-15:
                return None
            world.append(value[:3]/value[3])
        focal=np.asarray(self.plotter.camera.focal_point)
        if self._view_name=="front":
            origin,normal=focal,np.array([0.,1,0])
        elif self._view_name in ("right","side"):
            origin,normal=focal,np.array([1.,0,0])
        else:
            origin,normal=np.zeros(3),np.array([0.,0,1])
        return intersect_plane(world[0],world[1],origin,normal)
