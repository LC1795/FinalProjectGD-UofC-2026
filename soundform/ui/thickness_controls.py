"""Controls for converting generated meshes into frame members and solid tiles."""
from PySide6.QtWidgets import QWidget,QFormLayout,QLabel,QPushButton,QCheckBox
from soundform.geometry.mesh_thickness import delaunay_frame,voronoi_solids


class ThicknessControlsMixin:
    def _add_thickness_controls(self,form,method):
        if method not in ("delaunay","voronoi"):
            return
        if not hasattr(self,"_thickness_groups"):
            self._thickness_groups = {}
            self._thickness_inputs = {}
        host = QWidget()
        controls = QFormLayout(host)
        controls.setContentsMargins(0,8,0,0)
        note = QLabel("Generate the mesh above first, then choose uniform or sound-weighted sizes.")
        note.setWordWrap(True)
        controls.addRow(note)
        unit = "radius" if method=="delaunay" else "thickness"
        uniform = self._number(.05 if method=="delaunay" else .1,.001,100," m")
        lower = self._number(.02 if method=="delaunay" else .03,.001,100," m")
        upper = self._number(.15 if method=="delaunay" else .3,.001,100," m")
        if method=="delaunay":
            hint = QLabel("Each unique edge becomes a capped strut; connected points become spheres. Weighted struts taper between endpoint radii.")
        else:
            hint = QLabel("Tiles flatten to their centre-point tangent plane by default. Thickness is centred on that plane or on the curved surface.")
        hint.setWordWrap(True)
        controls.addRow(hint)
        curve = None
        if method=="voronoi":
            curve = QCheckBox("Preserve surface curvature")
            curve.setToolTip("Offset the existing curved tile along its vertex normals, then close its boundary walls.")
            controls.addRow(curve)
        controls.addRow("Uniform "+unit,uniform)
        one = QPushButton("Create uniform struts and spheres" if method=="delaunay" else "Create uniform solid tiles")
        one.clicked.connect(lambda checked=False,kind=method:self._thicken_mesh(kind,False))
        controls.addRow(one)
        controls.addRow("Minimum "+unit,lower)
        controls.addRow("Maximum "+unit,upper)
        two = QPushButton("Create sound-weighted struts and spheres" if method=="delaunay" else "Create sound-weighted solid tiles")
        two.clicked.connect(lambda checked=False,kind=method:self._thicken_mesh(kind,True))
        controls.addRow(two)
        form.addRow(host)
        self._thickness_groups[method] = host
        self._thickness_inputs[method] = (uniform,lower,upper,curve)
        host.setEnabled(False)

    def _update_thickness_controls(self):
        for method,host in getattr(self,"_thickness_groups",{}).items():
            host.setEnabled(method in self.viewport._reconstructions and not getattr(self,"_busy",False))

    def _thicken_mesh(self,method,weighted):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before adding thickness.")
            return
        original = self.viewport._reconstructions.get(method)
        points = self.viewport._cloud_points
        if original is None or points is None:
            self.statusBar().showMessage("Generate the "+method.title()+" mesh first.")
            return
        uniform,lower,upper,curve = self._thickness_inputs[method]
        size,minimum,maximum = uniform.value(),lower.value(),upper.value()
        keep_curve = curve.isChecked() if curve is not None else False
        snapshot,cloud = original.copy(),points.copy()
        boundary = self.viewport._cloud_surface.copy() if self.viewport._cloud_surface is not None else None
        parent_id,points_id = id(original),id(points)
        def job():
            if method=="delaunay":
                mesh = delaunay_frame(snapshot,cloud,radius=size,weighted=weighted,
                    min_radius=minimum,max_radius=maximum)
                target = "delaunay-frame"
            else:
                mesh = voronoi_solids(snapshot,cloud,boundary,thickness=size,weighted=weighted,
                    min_thickness=minimum,max_thickness=maximum,preserve_curvature=keep_curve)
                target = "voronoi-solid"
            return method,target,mesh,parent_id,points_id
        self._start("mesh-thickness",job)
