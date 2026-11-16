"""Interactive design view.

The sound sphere shows the recording's directional energy around the listener.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._params = DesignParameters()
        self._sphere_key = None
        self._ground_size = None
        self._site_model = None
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
        layout.addLayout(controls)
        self.caption = QLabel(
            f"Grid 1 m · Major lines 5 m · Microphone {EAR_HEIGHT_M:g} m above ground\n"
            "Right-drag orbit · Shift + right-drag pan · Wheel zoom\n"
            "Downward arrivals are shown at the horizon."
        )
        self.caption.setObjectName("hint")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
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
        colors = {"Terrain": "#687a62", "Solids": "#9ca7ad", "Vegetation": "#527345"}
        for category, color in colors.items():
            actor = f"context-{category}"
            self._drop(actor)
            if model is not None and category in model.meshes:
                vertices, faces = model.meshes[category]
                cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
                self._replace(actor, pv.PolyData(vertices, cells), color=color, reset_camera=False)
        self._drop("context-sources")
        if model is not None and model.sources:
            points = pv.PolyData(np.array([point for _, point in model.sources]))
            self._replace("context-sources", points, color="#e59355", style="points",
                          render_points_as_spheres=True, point_size=13, reset_camera=False)
        self._draw_ground(self._params)
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
                if event.button() in (Qt.LeftButton, Qt.MiddleButton):
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
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            self._drop("architecture")
        else:
            self._replace(
                "architecture",
                _polydata(mesh),
                color="#d9c7a6",
                smooth_shading=True,
                show_scalar_bar=False,
            )
        self._apply_visibility()
        self.plotter.render()

    def set_pavilion_selection(self, mesh) -> None:
        self._drop("pavilion-selection")
        if mesh is not None:
            self._replace("pavilion-selection", _polydata(mesh), color="#f4b95e",
                          style="wireframe", line_width=2, lighting=False, reset_camera=False)
        self._apply_visibility()
        self.plotter.render()

    def set_overlay(self, azimuth_deg: np.ndarray | None, elevation_deg: np.ndarray | None,
                    scores: np.ndarray | None, params: DesignParameters) -> None:
        self._params = params
        bars = getattr(self.plotter, "scalar_bars", {})
        for title in ("Energy", "Relative energy"):
            if title in bars:
                self.plotter.remove_scalar_bar(title)
        points = None
        if azimuth_deg is not None and elevation_deg is not None and scores is not None:
            points = _sphere_samples(azimuth_deg, elevation_deg, scores, params)
        center = (params.protected_center_x, params.protected_center_y, EAR_HEIGHT_M)
        sphere_key = (center, _ring_radius(params)) if azimuth_deg is not None else None
        if sphere_key != self._sphere_key:
            self._sphere_key = sphere_key
            self._drop("sound-sphere")
            if sphere_key is not None:
                self._replace(
                    "sound-sphere",
                    pv.Sphere(radius=sphere_key[1], center=center, theta_resolution=24, phi_resolution=9, end_phi=90),
                    style="wireframe", color="#b2bcb8", opacity=0.16, show_scalar_bar=False,
                )
        if points is None:
            self._drop("sound-field")
        else:
            self._replace(
                "sound-field",
                points,
                style="points",
                render_points_as_spheres=True,
                point_size=10,
                scalars="energy",
                cmap="magma",
                clim=(0.0, 1.0),
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Relative energy",
                    "color": "#e7e2d6",
                    "n_labels": 3,
                    "fmt": "%.1f",
                    "vertical": True,
                    "position_x": 0.84,
                    "position_y": 0.2,
                    "width": 0.05,
                    "height": 0.5,
                },
            )
        self._apply_visibility()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

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
        flags = {
            "architecture": self._show_geometry,
            "pavilion-selection": self._show_geometry,
            "sound-field": self._show_sound,
            "sound-sphere": self._show_sound,
            "zone": self._show_zone,
        }
        for name, visible in flags.items():
            actor = self.plotter.actors.get(name)
            if actor is not None:
                actor.SetVisibility(bool(visible))

    def _replace(self, name: str, mesh, **kwargs) -> None:
        self._drop(name)
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
