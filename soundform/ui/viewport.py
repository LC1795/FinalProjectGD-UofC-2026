"""Interactive design view.

The site, occupant zone, and compass are an abstract design frame. Coloured
columns show the recording's directional energy in that frame. They are not a
map of the place where the microphone stood.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from soundform.geometry.design_space import characteristic_radius, sound_az_el_to_design_vector
from soundform.models.project import DesignParameters


def _polydata(mesh) -> pv.PolyData:
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int64)
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces.ravel())


def _ring_radius(params: DesignParameters) -> float:
    """Place the sound halo just outside the site so the form does not hide it."""
    if params.site_shape == "rectangle":
        return 0.5 * float(np.hypot(params.site_width, params.site_depth)) * 1.06
    return max(2.0, float(params.site_radius) * 1.12)


def _columns(azimuth_deg: np.ndarray, scores: np.ndarray, params: DesignParameters) -> pv.PolyData | None:
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    keep = scores >= 0.12
    if not np.any(keep):
        return None
    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float64)[keep]
    scores = scores[keep]
    radius = _ring_radius(params)
    design_x, design_y, _up = sound_az_el_to_design_vector(
        azimuth_deg, np.zeros_like(azimuth_deg), params.rotation_offset_deg
    )
    horizontal = np.maximum(np.hypot(design_x, design_y), 1e-8)
    unit_x = design_x / horizontal
    unit_y = design_y / horizontal
    center_x = params.protected_center_x + unit_x * radius
    center_y = params.protected_center_y + unit_y * radius
    height = 0.35 + scores * max(1.5, float(params.max_height) * 0.85)
    half = max(0.35, radius * 0.018)
    tangent_x = -unit_y
    tangent_y = unit_x
    vertices = []
    faces = []
    for index in range(len(azimuth_deg)):
        base = len(vertices)
        cx = center_x[index]
        cy = center_y[index]
        top = height[index]
        offsets = (
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, 0.0),
            (-half, half, 0.0),
            (-half, -half, top),
            (half, -half, top),
            (half, half, top),
            (-half, half, top),
        )
        for ox, oy, oz in offsets:
            vertices.append(
                (
                    cx + ox * tangent_x[index] + oy * unit_x[index],
                    cy + ox * tangent_y[index] + oy * unit_y[index],
                    oz,
                )
            )
        quads = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        for a, b, c, d in quads:
            faces.append((base + a, base + b, base + c))
            faces.append((base + a, base + c, base + d))
    poly = pv.PolyData(
        np.asarray(vertices, dtype=np.float64),
        np.hstack([np.full((len(faces), 1), 3), np.asarray(faces, dtype=np.int64)]).ravel(),
    )
    # Two triangles share each face of the box; paint both with the same score.
    cell_scores = np.repeat(np.clip(scores, 0.0, 1.0), 12)
    poly.cell_data["energy"] = cell_scores[: poly.n_cells]
    return poly


class Viewport(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._params = DesignParameters()
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
            "Abstract site. Green is design forward (+Y). Columns are the recorded sound field, not a map."
        )
        self.caption.setObjectName("hint")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1c1e1c")
        self.plotter.add_axes(interactive=False, line_width=2, labels_off=False)
        layout.addWidget(self.plotter.interactor, stretch=1)
        self._camera_ready = False
        self.set_guides(self._params)

    def set_guides(self, params: DesignParameters) -> None:
        self._params = params
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

    def set_overlay(self, azimuth_deg: np.ndarray | None, scores: np.ndarray | None, params: DesignParameters) -> None:
        self._params = params
        bars = getattr(self.plotter, "scalar_bars", {})
        for title in ("Energy", "Relative energy"):
            if title in bars:
                self.plotter.remove_scalar_bar(title)
        columns = None
        if azimuth_deg is not None and scores is not None and len(azimuth_deg) > 0:
            columns = _columns(np.asarray(azimuth_deg, dtype=np.float64), np.asarray(scores, dtype=np.float64), params)
        if columns is None:
            self._drop("sound-field")
        else:
            self._replace(
                "sound-field",
                columns,
                scalars="energy",
                cmap="magma",
                clim=(0.0, 1.0),
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Energy",
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
        self.plotter.render()

    def set_view(self, name: str) -> None:
        radius = max(8.0, characteristic_radius(self._params))
        distance = radius * 3.1
        center = (0.0, 0.0, max(1.0, self._params.max_height * 0.25))
        if name == "top":
            position = (center[0], center[1], distance)
            up = (0.0, 1.0, 0.0)
        elif name == "front":
            position = (0.0, distance, distance * 0.45)
            up = (0.0, 0.0, 1.0)
        elif name == "side":
            position = (distance, 0.0, distance * 0.45)
            up = (0.0, 0.0, 1.0)
        else:
            position = (distance * 0.85, -distance * 0.95, distance * 0.72)
            up = (0.0, 0.0, 1.0)
        self.plotter.camera_position = [position, center, up]
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
            "sound-field": self._show_sound,
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
