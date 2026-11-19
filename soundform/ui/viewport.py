"""Interactive design view.

The sound sphere shows the recording's directional energy around the listener.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal, QEvent, Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from soundform.geometry.analysis_envelope import merged_envelope, envelope_energy, analyze_envelope, clip_to_terrain, scatter_surface_points
from soundform.geometry.design_space import EAR_HEIGHT_M, characteristic_radius, sound_az_el_to_design_vector
from soundform.models.project import DesignParameters


def _polydata(mesh) -> pv.PolyData:
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces.ravel())


def _ring_radius(params: DesignParameters) -> float:
    """Place the sound halo just outside the site so the form does not hide it."""
    if params.site_shape == "rectangle":
        return 0.5 * float(np.hypot(params.site_width, params.site_depth)) * 1.06
    return max(2.0, float(params.site_radius) * 1.12)


def _sphere_samples(azimuth_deg, elevation_deg, scores, params) -> pv.PolyData | None:
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    # Preserve the bearing of downward arrivals without placing markers underground.
    above = elevation_deg > 0
    horizon = ~above
    if np.any(horizon):
        scores = np.vstack((scores[horizon].max(axis=0), scores[above]))
        elevation_deg = np.concatenate(([0.0], elevation_deg[above]))
    keep = scores >= 0.06
    if not np.any(keep):
        return None
    azimuth, elevation = np.meshgrid(azimuth_deg, elevation_deg)
    directions = sound_az_el_to_design_vector(azimuth[keep], elevation[keep], params.rotation_offset_deg)
    center = (params.protected_center_x, params.protected_center_y, EAR_HEIGHT_M)
    points = pv.PolyData(np.column_stack(directions) * _ring_radius(params) + center)
    points.point_data["energy"] = scores[keep]
    return points


class Viewport(QWidget):
    object_selected = Signal(object)
    envelope_updated = Signal()
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._params = DesignParameters()
        self._sphere_key = None
        self._ground_size = None
        self._terrain = None
        self._site_model = None
        self._pick_objects = {}
        self._focus_points = None
        self._navigation_scale = 1.0
        self._envelope_key = None
        self._envelope = None
        self._sampled_envelope = None
        self._metaball_enabled = False
        self._cloud_members = None
        self._cloud_surface = None
        self._cloud_sampled = None
        self._cloud_points = None
        self._reconstructions = {}
        self._layer_visibility = {}
        self._object_visibility = {}
        self._pavilion_source = None
        self._selection_source = None
        self._cloud_strength = 1.0
        self._cloud_radius_scale = 1.0
        self._cloud_opacity = 0.28
        self._selected_origin_indices = set()
        self._origin_settings = {}
        self._overlay_input = None
        self._drag = None
        self._view_name = "perspective"
        self._show_geometry = True
        self._show_sound = True
        self._show_zone = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.geometry_box = QCheckBox("Architecture")
        self.sound_box = QCheckBox("Sound field")
        self.zone_box = QCheckBox("Occupant zone")
        for box, slot in (
            (self.geometry_box, self._toggle_geometry),
            (self.sound_box, self._toggle_sound),
            (self.zone_box, self._toggle_zone),
        ):
            box.setChecked(True)
            box.toggled.connect(slot)
            controls.addWidget(box)
        controls.addStretch(1)
        for label, name in (("Top", "top"), ("Front", "front"), ("Side", "side"), ("Perspective", "perspective")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, view=name: self.set_view(view))
            controls.addWidget(button)
        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda *_args: self.reset_view())
        controls.addWidget(reset)
        self.view_controls = QWidget()
        view_layout = QVBoxLayout(self.view_controls)
        view_layout.addLayout(controls)
        self.caption = QLabel(
            f"Grid 1 m · Major lines 5 m · Microphone {EAR_HEIGHT_M:g} m above ground\n"
            "Middle-click frame object • Right-drag orbit · Shift + right-drag pan · Wheel zoom\n"
            "Downward arrivals are shown at the horizon."
        )
        self.caption.setObjectName("hint")
        self.caption.setWordWrap(True)
        view_layout.addWidget(self.caption)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1c1e1c")
        self.plotter.add_axes(interactive=False, line_width=2, labels_off=False)
        layout.addWidget(self.plotter.interactor, stretch=1)
        self.plotter.interactor.installEventFilter(self)
        self._camera_ready = False
        self.set_guides(self._params)

    def set_guides(self, params: DesignParameters) -> None:
        self._params = params
        self._draw_ground(params)
        self._draw_microphone(params)
        self._replace("site", self._site_mesh(params), color="#9aa194", line_width=2)
        zone = pv.Disc(
            center=(params.protected_center_x, params.protected_center_y, 0.05),
            inner=0.0,
            outer=max(0.4, params.protected_radius),
            normal=(0, 0, 1),
            c_res=64,
        )
        self._replace("zone", zone, color="#c4a574", opacity=0.38)
        self._compass(params)
        self._apply_visibility()
        if not self._camera_ready:
            self.set_view("perspective")
            self._camera_ready = True
        else:
            self.plotter.render()

    def set_site_model(self, model) -> None:
        self._site_model = model
        self._origin_settings = {}
        self._cloud_members = None
        self._metaball_enabled = False
        self._cloud_points = None
        self._drop("cloud-points")
        colors = {"Terrain": "#687a62", "Solids": "#9ca7ad", "Vegetation": "#527345"}
        for name in list(self._pick_objects):
            self._drop(name)
        self._pick_objects.clear()
        for category in colors:
            self._drop(f"context-{category}")
        if model is not None:
            for index, item in enumerate(model.objects):
                name = f"context-object-{index}"
                cells = np.column_stack((np.full(len(item["faces"]), 3), item["faces"])).ravel()
                self._replace(name, pv.PolyData(item["vertices"], cells),
                              color=colors[item["category"]], reset_camera=False)
                self._pick_objects[name] = item
            if not model.objects:
                for category, (vertices, faces) in model.meshes.items():
                    cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
                    self._replace(f"context-{category}", pv.PolyData(vertices, cells),
                                  color=colors[category], reset_camera=False)
            for index, (label, point) in enumerate(model.origins):
                name = f"origin-{index}"
                self._replace(name, pv.Sphere(center=point, radius=0.22),
                              color="#f2bc65", reset_camera=False)
                self._pick_objects[name] = {"name": label, "category": "Pavilion Origin",
                                            "point": point, "origin_index": index}
        self._drop("context-sources")
        if model is not None:
            for index, (label, point) in enumerate(model.sources):
                name = f"source-{index}"
                self._replace(name, pv.Sphere(center=point, radius=0.18),
                              color="#e59355", reset_camera=False)
                self._pick_objects[name] = {"name": label, "category": "Sound Sources", "point": point}
        self._terrain = None
        if model is not None and "Terrain" in model.meshes:
            vertices, faces = model.meshes["Terrain"]
            self._terrain = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        self._envelope_key = None
        self._draw_ground(self._params)
        if self._overlay_input is not None:
            self.set_overlay(*self._overlay_input, self._params)
        self.plotter.render()

    def _draw_ground(self, params: DesignParameters) -> None:
        size = ((params.site_width, params.site_depth) if params.site_shape == "rectangle"
                else (2 * params.site_radius, 2 * params.site_radius))
        if self._site_model is not None and "Terrain" in self._site_model.meshes:
            for actor in ("ground", "grid-major", "grid-minor"):
                self._drop(actor)
            self._ground_size = None
            return
        if self._ground_size == size:
            return
        self._ground_size = size
        width, depth = size
        self._replace("ground", pv.Plane(center=(0, 0, 0), i_size=width, j_size=depth, i_resolution=1,
                                        j_resolution=1), color="#46504a", opacity=0.22, lighting=False)
        for spacing, name, color, opacity in ((1, "minor", "#67756d", 0.34), (5, "major", "#a0aaa1", 0.56)):
            hx, hy = width / 2, depth / 2
            ix, iy = int(hx // spacing), int(hy // spacing)
            x_lines = [(i * spacing, y, 0.005) for i in range(-ix, ix + 1)
                       if spacing == 5 or i % 5 != 0 for y in (-hy, hy)]
            y_lines = [(x, i * spacing, 0.005) for i in range(-iy, iy + 1)
                       if spacing == 5 or i % 5 != 0 for x in (-hx, hx)]
            points = x_lines + y_lines
            if points:
                self._replace(f"grid-{name}", pv.line_segments_from_points(np.asarray(points)),
                              color=color, opacity=opacity)
            else:
                self._drop(f"grid-{name}")

    def _draw_microphone(self, params: DesignParameters) -> None:
        x, y = params.protected_center_x, params.protected_center_y
        center = (x, y, EAR_HEIGHT_M)
        self._replace("microphone", pv.Sphere(radius=0.14, center=center), color="#efca88")
        self._replace("mic-height", pv.Line((x, y, 0.02), center), color="#efca88", line_width=2)

    def eventFilter(self, watched, event):
        if watched is self.plotter.interactor:
            if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
                # Keep VTK's default trackball gestures from competing with right-drag navigation.
                if event.button() == Qt.LeftButton:
                    if event.type() == QEvent.MouseButtonPress:
                        self._pick_object(event.position())
                    return True
                if event.button() == Qt.MiddleButton:
                    if event.type() == QEvent.MouseButtonPress:
                        self.focus_object(event.position())
                    return True
            if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick) and event.button() == Qt.RightButton:
                modifiers = event.modifiers()
                if modifiers & Qt.ControlModifier:
                    mode = "zoom"
                elif modifiers & Qt.ShiftModifier or self._view_name != "perspective":
                    mode = "pan"
                else:
                    mode = "orbit"
                self._drag = (mode, event.position())
                return True
            if event.type() == QEvent.MouseMove and self._drag is not None:
                mode, previous = self._drag
                delta = event.position() - previous
                self._drag = (mode, event.position())
                self._navigate(mode, delta.x(), delta.y())
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                self._drag = None
                return True
        return super().eventFilter(watched, event)

    def _pick_object(self, position, *, clear=True):
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        picker = vtkCellPicker()
        picker.SetTolerance(0.008)
        picker.PickFromListOn()
        names = ["architecture", "cloud-points", *self._pick_objects, *[name for name in self.plotter.actors if name.startswith(("analysis-sphere-", "reconstruction-"))], "sound-field"]
        for name in names:
            actor = self.plotter.actors.get(name)
            if name != "sound-field" and actor is not None and actor.GetVisibility():
                picker.AddPickList(actor)
        ratio = self.plotter.interactor.devicePixelRatioF()
        x = position.x() * ratio
        y = (self.plotter.interactor.height() - position.y()) * ratio - 1
        picker.Pick(x, y, 0, self.plotter.renderer)
        actor = picker.GetActor()
        if actor is None:
            envelope_actor = self.plotter.actors.get("sound-field")
            if envelope_actor is not None and envelope_actor.GetVisibility():
                picker.AddPickList(envelope_actor)
                picker.Pick(x, y, 0, self.plotter.renderer)
                actor = picker.GetActor()
        for name in names:
            if actor is not None and actor == self.plotter.actors.get(name):
                if name == "architecture":
                    self.object_selected.emit({"category": "Pavilion", "cell": int(actor.mapper.dataset.cell_data["source_cell"][picker.GetCellId()])})
                elif name == "cloud-points":
                    self.object_selected.emit({"category": "Surface points", "name": "Surface points"})
                elif name.startswith("analysis-sphere-"):
                    self.object_selected.emit({"category": "Analysis sphere", "name": "Analysis sphere",
                                               "origin_index": int(name.rsplit("-", 1)[1]),
                                               "extend_selection": bool(QApplication.keyboardModifiers() & Qt.ControlModifier)})
                elif name.startswith("reconstruction-"):
                    mesh = self._reconstructions[name.removeprefix("reconstruction-")]
                    self.object_selected.emit({"category": "Reconstruction", "name": name.removeprefix("reconstruction-").title() + " mesh",
                                               "vertices": mesh.points, "faces": mesh.faces.reshape(-1, 4)[:, 1:]})
                elif name == "sound-field":
                    self.object_selected.emit({"category": "Analysis envelope", "name": "Analysis envelope"})
                else:
                    item = dict(self._pick_objects[name])
                    item["extend_selection"] = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
                    self.object_selected.emit(item)
                return
        if clear:
            self.object_selected.emit(None)

    def focus_object(self, position=None):
        if position is not None:
            self._pick_object(position, clear=False)
        points = self._focus_points
        if points is None or not len(points):
            return
        camera = self.plotter.camera
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        direction = np.asarray(camera.position) - np.asarray(camera.focal_point)
        direction /= max(np.linalg.norm(direction), 1e-8)
        right = np.cross(np.asarray(camera.up), direction)
        right /= max(np.linalg.norm(right), 1e-8)
        up = np.cross(direction, right)
        relative = points - center
        horizontal = np.abs(relative @ right)
        vertical = np.abs(relative @ up)
        depth = relative @ direction
        self._navigation_scale = max(float(np.ptp(points, axis=0).max()), 0.05)
        width, height = self.plotter.interactor.width(), self.plotter.interactor.height()
        aspect = width / max(height, 1)
        # Fit projected vertices inside both the vertical and sloping hexagon edges.
        required = np.maximum(vertical + horizontal / np.sqrt(3),
                              horizontal / max(aspect, 1e-6))
        half_height = max(float(required.max()) * 1.04, self._navigation_scale * 0.001)
        if camera.parallel_projection:
            camera.parallel_scale = half_height
            distance = max(self._navigation_scale * 2, np.linalg.norm(direction))
        else:
            tangent = np.tan(np.deg2rad(camera.view_angle / 2))
            distance = max(float((depth + required * 1.04 / tangent).max()),
                           self._navigation_scale * 0.01)
        camera.focal_point = center
        camera.position = center + direction * distance
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _navigate(self, mode: str, dx: float, dy: float) -> None:
        camera = self.plotter.camera
        target = np.asarray(camera.focal_point)
        offset = np.asarray(camera.position) - target
        distance = np.linalg.norm(offset)
        if mode == "orbit":
            bearing = np.arctan2(offset[1], offset[0]) - dx * 0.006
            elevation = np.arctan2(offset[2], np.hypot(*offset[:2])) + dy * 0.006
            elevation = np.clip(elevation, -np.deg2rad(89), np.deg2rad(89))
            horizontal = np.cos(elevation)
            direction = np.array([horizontal * np.cos(bearing), horizontal * np.sin(bearing), np.sin(elevation)])
            camera.position = target + distance * direction
            camera.up = (0, 0, 1)
        elif mode == "pan":
            normal = offset / distance
            right = np.cross(camera.up, normal)
            right /= np.linalg.norm(right)
            up = np.cross(normal, right)
            half_height = (camera.parallel_scale if camera.parallel_projection
                           else distance * np.tan(np.deg2rad(camera.view_angle / 2)))
            shift = (2 * half_height / max(1, self.plotter.interactor.height())) * (-dx * right + dy * up)
            camera.position = target + offset + shift
            camera.focal_point = target + shift
        else:
            factor = np.exp(np.clip(-dy * 0.01, -2, 2))
            if camera.parallel_projection:
                camera.parallel_scale /= factor
            else:
                camera.position = target + offset / factor
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def set_mesh(self, mesh) -> None:
        self._pavilion_source = mesh
        mesh = self._visible_pavilion(mesh)
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            self._drop("architecture")
        else:
            self._replace(
                "architecture",
                self._pavilion_polydata(mesh),
                color="#d9c7a6",
                smooth_shading=True,
                show_scalar_bar=False,
            )
        self._apply_visibility()
        self.plotter.render()

    def set_pavilion_selection(self, mesh) -> None:
        self._selection_source = mesh
        mesh = self._visible_pavilion(mesh)
        self._drop("pavilion-selection")
        if mesh is not None:
            self._replace("pavilion-selection", _polydata(mesh), color="#f4b95e",
                          style="wireframe", line_width=2, lighting=False, reset_camera=False)
        self._focus_points = None if mesh is None else np.asarray(mesh.vertices)
        self._apply_visibility()
        self.plotter.render()

    def set_overlay(self, azimuth_deg: np.ndarray | None, elevation_deg: np.ndarray | None,
                    scores: np.ndarray | None, params: DesignParameters) -> None:
        self._params = params
        bars = getattr(self.plotter, "scalar_bars", {})
        for title in ("Energy", "Relative energy"):
            if title in bars:
                self.plotter.remove_scalar_bar(title)
        self._overlay_input = (azimuth_deg, elevation_deg, scores)
        self._drop("sound-sphere")
        self._drop("sound-field")
        self._sampled_envelope = None
        for name in list(self.plotter.actors):
            if name.startswith(("analysis-sphere-", "analysis-fill-")):
                self._drop(name)
        has_audio = azimuth_deg is not None and elevation_deg is not None and scores is not None
        centers = []
        if self._site_model is not None:
            centers = [point + (0, 0, EAR_HEIGHT_M) for _, point in self._site_model.origins]
        if not centers:
            centers = [(params.protected_center_x, params.protected_center_y, EAR_HEIGHT_M)]
        centers = np.asarray(centers, dtype=float)
        radii = [self._origin_settings.get(i, {}).get("radius", max(0.5, params.pavilion_diameter / 2))
                 for i in range(len(centers))]
        members = tuple(i for i in (self._cloud_members or ()) if i < len(centers))
        for index in members:
            radii[index] *= self._cloud_radius_scale
        key = (tuple(centers.ravel()), tuple(radii), members,
               self._metaball_enabled, self._cloud_strength)
        if key != self._envelope_key:
            self._envelope_key = key
            self._cloud_surface = None
            self._cloud_sampled = None
            self._cloud_points = None
            self._drop("cloud-points")
            self.clear_reconstructions()
            pieces = []
            if self._metaball_enabled and members:
                self._cloud_surface = clip_to_terrain(
                    merged_envelope(centers[list(members)], np.array(radii)[list(members)],
                                    strength=self._cloud_strength, full_sphere=True), self._terrain)
                pieces.append(self._cloud_surface)
            for i, center in enumerate(centers):
                if self._metaball_enabled and i in members:
                    continue
                sphere = clip_to_terrain(merged_envelope([center], radii[i], merge=False, full_sphere=True), self._terrain)
                pieces.append(sphere)
            self._envelope = pieces[0].merge(pieces[1:], merge_points=False) if len(pieces) > 1 else pieces[0]
        for i, center in enumerate(centers):
            if self._metaball_enabled and i in members:
                continue
            sphere = clip_to_terrain(merged_envelope([center], radii[i], merge=False, full_sphere=True), self._terrain)
            sphere.point_data["energy"] = (
                envelope_energy(sphere.points, [center], azimuth_deg, elevation_deg, scores,
                                params.rotation_offset_deg) if has_audio else np.zeros(sphere.n_points))
            self._replace(f"analysis-fill-{i}", sphere, scalars="energy", cmap="magma",
                          clim=(0, 1), opacity=self._cloud_opacity, show_scalar_bar=False)
            self._replace(f"analysis-sphere-{i}", sphere, style="wireframe",
                          color="#b2bcb8", opacity=0.16, show_scalar_bar=False)
        self._cloud_sampled = None
        if self._cloud_surface is not None and has_audio and self._cloud_surface.n_points:
            self._cloud_sampled = self._cloud_surface.copy()
            self._cloud_sampled.point_data["energy"] = envelope_energy(
                self._cloud_sampled.points, centers[list(members)], azimuth_deg, elevation_deg,
                scores, params.rotation_offset_deg)
        if self._cloud_points is not None and has_audio:
            self._cloud_points.point_data["energy"] = envelope_energy(
                self._cloud_points.points, centers[list(members)], azimuth_deg, elevation_deg,
                scores, params.rotation_offset_deg)
            self._draw_cloud_points()
        surface = self._envelope.copy()
        if surface.n_points:
            surface.point_data["energy"] = (
                envelope_energy(surface.points, centers, azimuth_deg, elevation_deg, scores,
                                params.rotation_offset_deg) if has_audio else np.zeros(surface.n_points)
            )
            self._sampled_envelope = surface if has_audio else None
            displayed = self._cloud_sampled if self._cloud_sampled is not None else self._cloud_surface
            if displayed is None:
                displayed = pv.PolyData()
            elif "energy" not in displayed.point_data:
                displayed = displayed.copy()
                displayed.point_data["energy"] = np.zeros(displayed.n_points)
            if displayed.n_points:
                self._replace(
                "sound-field", displayed, scalars="energy", cmap="magma", clim=(0, 1),
                opacity=self._cloud_opacity, smooth_shading=True, show_scalar_bar=has_audio,
                scalar_bar_args={"title": "Relative energy", "color": "#e7e2d6",
                                 "n_labels": 3, "vertical": True, "position_x": 0.84,
                                 "position_y": 0.35, "width": 0.05, "height": 0.30},
            )
                self._replace("sound-sphere", displayed, style="wireframe",
                              color="#b2bcb8", opacity=0.09, show_scalar_bar=False)
        self._apply_visibility()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        self.set_origin_selection(self._selected_origin_indices)
        self.envelope_updated.emit()

    def _draw_cloud_points(self):
        self._drop("cloud-points")
        if self._cloud_points is not None and self._cloud_points.n_points:
            self._replace("cloud-points", self._cloud_points, style="points",
                          render_points_as_spheres=True, point_size=5, scalars="energy",
                          cmap="magma", clim=(0, 1), show_scalar_bar=False)

    def generate_cloud_points(self, quiet_density, loud_density, energy_threshold=0.0):
        if self._cloud_sampled is None:
            raise ValueError("Create a metaball and analyze audio before generating surface points.")
        self._cloud_points, capped = scatter_surface_points(
            self._cloud_sampled, quiet_density, loud_density, energy_threshold=energy_threshold)
        self.clear_reconstructions()
        self._draw_cloud_points()
        self._apply_visibility()
        self.plotter.render()
        return self._cloud_points.n_points, capped

    def clear_reconstructions(self):
        for method in self._reconstructions:
            self._drop(f"reconstruction-{method}")
        self._reconstructions.clear()

    def show_reconstruction(self, method, mesh, color_by_cell=False):
        self._reconstructions[method] = mesh
        options = (dict(scalars="voronoi_cell", preference="cell", cmap="tab20")
                   if color_by_cell and "voronoi_cell" in mesh.cell_data
                   else dict(scalars="energy", cmap="magma", clim=(0, 1)))
        self._replace(f"reconstruction-{method}", mesh, **options,
                      opacity=0.95 if method == "voronoi" else 0.85, show_scalar_bar=False)
        self._apply_visibility()
        self.plotter.render()

    def object_visibility_key(self, item):
        category = item.get("category")
        if category == "Pavilion":
            return "part:" + item["name"]
        if category == "Analysis sphere":
            return "sphere:" + str(item["origin_index"])
        if category == "Analysis envelope":
            return "metaball"
        if category == "Surface points":
            return "cloud-points"
        if category == "Reconstruction":
            return "reconstruction-" + item["name"].removesuffix(" mesh").lower()
        for actor_name, obj in self._pick_objects.items():
            if obj is item:
                return actor_name
            if (obj.get("category") == category and obj.get("name") == item.get("name")
                    and obj.get("origin_index") == item.get("origin_index")
                    and np.array_equal(obj.get("point"), item.get("point"))):
                return actor_name
        return str(category) + ":" + item.get("name", "")

    def _visible_pavilion(self, mesh):
        if mesh is None:
            return None
        from soundform.geometry.pavilion_edit import part_labels
        mask = np.array([self._object_visibility.get("part:" + str(label), True)
                         for label in part_labels(mesh)])
        if not mask.any():
            return None
        visible = mesh.copy()
        visible.face_attributes["source_cell"] = np.arange(len(mesh.faces))
        visible.update_faces(mask)
        return visible

    def _pavilion_polydata(self, mesh):
        poly = _polydata(mesh)
        poly.cell_data["source_cell"] = mesh.face_attributes.get("source_cell", np.arange(len(mesh.faces)))
        return poly

    def set_object_visible(self, key, visible):
        self._object_visibility[key] = visible
        if key.startswith("part:"):
            self.set_mesh(self._pavilion_source)
            self.set_pavilion_selection(self._selection_source)
        self._apply_visibility()
        self.plotter.render()

    def _actor_object_key(self, name):
        if name.startswith(("analysis-sphere-", "analysis-fill-")):
            return "sphere:" + name.rsplit("-", 1)[1]
        if name in ("sound-field", "sound-sphere"):
            return "metaball"
        return name

    def set_layer_visible(self, layer, visible):
        self._layer_visibility[layer] = visible
        self._apply_visibility()
        self.plotter.render()

    def _actor_layer(self, name):
        if name in self._pick_objects:
            item = self._pick_objects[name]
            return item.get("layer", item["category"])
        if name.startswith(("analysis-sphere-", "analysis-fill-")):
            return "Analysis spheres"
        if name.startswith("reconstruction-"):
            return name.removeprefix("reconstruction-").title() + " mesh"
        if name in ("cloud-points", "sound-field", "sound-sphere"):
            return "Cloud" if self._metaball_enabled else "Analysis spheres"
        if name == "architecture":
            return "Pavilion"
        return None

    def set_origin_selection(self, indices):
        self._selected_origin_indices = set(indices)
        for index in range(len(self._site_model.origins) if self._site_model else 0):
            selected = index in self._selected_origin_indices
            for prefix in ("origin-", "analysis-sphere-"):
                actor = self.plotter.actors.get(f"{prefix}{index}")
                if actor is not None:
                    color = "#ffd26a" if selected else ("#f2bc65" if prefix == "origin-" else "#b2bcb8")
                    actor.prop.color = color
                    actor.prop.line_width = 2 if selected else 1
        self.plotter.render()

    def set_metaball_enabled(self, enabled):
        self._metaball_enabled = enabled
        if self._overlay_input is not None:
            self.set_overlay(*self._overlay_input, self._params)

    def surface_analysis(self, threshold=-12):
        if self._sampled_envelope is None:
            return None
        result = analyze_envelope(self._sampled_envelope, threshold)
        if result is not None:
            result["origin_properties"] = self._origin_settings
            result["connection_strength"] = self._cloud_strength
            result["cloud_origin_indices"] = list(self._cloud_members or ())
        return result

    def set_view(self, name: str) -> None:
        radius = max(8.0, characteristic_radius(self._params))
        distance = max(radius * 3.1, _ring_radius(self._params) * 3.0)
        self._view_name = name
        x, y = self._params.protected_center_x, self._params.protected_center_y
        center = (x, y, EAR_HEIGHT_M)
        if name == "top":
            position = (x, y, EAR_HEIGHT_M + distance)
            up = (0.0, 1.0, 0.0)
        elif name == "front":
            position = (x, y + distance, EAR_HEIGHT_M)
            up = (0.0, 0.0, 1.0)
        elif name == "side":
            position = (x + distance, y, EAR_HEIGHT_M)
            up = (0.0, 0.0, 1.0)
        else:
            position = (x + distance * 0.85, y - distance * 0.95, EAR_HEIGHT_M + distance * 0.72)
            up = (0.0, 0.0, 1.0)
        self.plotter.camera_position = [position, center, up]
        if name == "perspective":
            self.plotter.disable_parallel_projection()
        else:
            self.plotter.enable_parallel_projection()
            self.plotter.camera.parallel_scale = 1.15 * _ring_radius(self._params)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def reset_view(self) -> None:
        self.geometry_box.setChecked(True)
        self.sound_box.setChecked(True)
        self.zone_box.setChecked(True)
        self.set_view("perspective")

    def _toggle_geometry(self, checked: bool) -> None:
        self._show_geometry = checked
        self._apply_visibility()
        self.plotter.render()

    def _toggle_sound(self, checked: bool) -> None:
        self._show_sound = checked
        self._apply_visibility()
        self.plotter.render()

    def _toggle_zone(self, checked: bool) -> None:
        self._show_zone = checked
        self._apply_visibility()
        self.plotter.render()

    def _apply_visibility(self) -> None:
        flags = {name: self._show_sound for name in self.plotter.actors if name.startswith("analysis-sphere-")}
        flags.update({
            "architecture": self._show_geometry,
            "pavilion-selection": self._show_geometry,
            "sound-field": self._show_sound,
            "sound-sphere": self._show_sound,
            "zone": self._show_zone,
            "cloud-points": self._show_sound and self._metaball_enabled,
        })
        for name, visible in flags.items():
            actor = self.plotter.actors.get(name)
            if actor is not None:
                actor.SetVisibility(bool(visible))
        for name, actor in self.plotter.actors.items():
            layer = self._actor_layer(name)
            if layer is not None:
                visible = self._layer_visibility.get(layer, True)
                if layer in ("Cloud", "Analysis spheres") or name.startswith("reconstruction-"):
                    visible = visible and self._show_sound
                if name == "cloud-points":
                    visible = visible and self._metaball_enabled
                if name == "architecture":
                    visible = visible and self._show_geometry
                visible = visible and self._object_visibility.get(self._actor_object_key(name), True)
                actor.SetVisibility(visible)
        selection = self.plotter.actors.get("pavilion-selection")
        if selection is not None and not self._layer_visibility.get("Pavilion", True):
            selection.SetVisibility(False)

    def _replace(self, name: str, mesh, **kwargs) -> None:
        self._drop(name)
        kwargs.setdefault('reset_camera', False)
        self.plotter.add_mesh(mesh, name=name, **kwargs)

    def _drop(self, name: str) -> None:
        if name in self.plotter.actors:
            self.plotter.remove_actor(name)

    def _site_mesh(self, params: DesignParameters) -> pv.PolyData:
        if params.site_shape == "rectangle":
            half_x = params.site_width * 0.5
            half_y = params.site_depth * 0.5
            points = np.array(
                [[-half_x, -half_y, 0.02], [half_x, -half_y, 0.02], [half_x, half_y, 0.02], [-half_x, half_y, 0.02]],
                dtype=np.float64,
            )
        else:
            angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
            points = np.column_stack(
                (
                    np.cos(angles) * params.site_radius,
                    np.sin(angles) * params.site_radius,
                    np.full(angles.shape, 0.02),
                )
            )
        closed = np.vstack([points, points[:1]])
        return pv.lines_from_points(closed)

    def _compass(self, params: DesignParameters) -> None:
        length = max(3.0, characteristic_radius(params) * 0.22)
        origin = np.array([params.protected_center_x, params.protected_center_y, 0.08])
        axes = (
            ("axis-x", (1.0, 0.0, 0.0), "#c45c4a"),
            ("axis-y", (0.0, 1.0, 0.0), "#7d9a62"),
            ("axis-z", (0.0, 0.0, 1.0), "#7f92b5"),
        )
        for name, direction, color in axes:
            end = origin + np.array(direction) * length
            self._replace(name, pv.Line(origin, end), color=color, line_width=4)

    def shutdown(self) -> None:
        self.plotter.close()
