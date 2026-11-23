"""Cloud tables, geometric point filters, and editable origin positions."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QPushButton, QLabel,
    QToolButton)
from soundform.geometry.point_filters import subtract_curve_tubes


class CloudControlsMixin:
    def _cloud_section(self, layout, title):
        toggle = QToolButton()
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.RightArrow)
        content = QWidget()
        body = QVBoxLayout(content)
        content.hide()
        toggle.toggled.connect(lambda opened: (content.setVisible(opened),
            toggle.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)))
        layout.addWidget(toggle)
        layout.addWidget(content)
        return body

    def _table(self, headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setMinimumHeight(160)
        return table

    def _number(self, value, minimum=0, maximum=1000, suffix=""):
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(.1)
        box.setSuffix(suffix)
        box.setKeyboardTracking(False)
        return box

    def _build_cloud_controls(self):
        cloud = QWidget()
        layout = QVBoxLayout(cloud)
        self.cloud_selection = QLabel("Choose metaball members in the origin table below.")
        self.cloud_selection.setWordWrap(True)
        layout.addWidget(self.cloud_selection)
        body = self._cloud_section(layout, "Metaball")
        self.cloud_origins = self._table(["Origin", "Include", "Radius scale"])
        body.addWidget(self.cloud_origins)
        form = QFormLayout()
        self.cloud_strength = self._number(1, .1, 3)
        form.addRow("Connection strength", self.cloud_strength)
        # Kept internally for compatibility; radius scaling is now per origin.
        self.cloud_radius_scale = self._number(1, .25, 4)
        self.cloud_radius_scale.hide()
        self.cloud_opacity = self._number(.28, .05, 1)
        form.addRow("Opacity", self.cloud_opacity)
        body.addLayout(form)
        self.create_cloud = QPushButton("Generate metaball")
        body.addWidget(self.create_cloud)
        body = self._cloud_section(layout, "Points")
        form = QFormLayout()
        self.cloud_quiet_density = self._number(.2, 0, 100, " points/m^2")
        self.cloud_loud_density = self._number(4, 0, 100, " points/m^2")
        self.cloud_energy_threshold = self._number(0, 0, 1)
        self.cloud_merge_distance = self._number(0, 0, 1000, " m")
        self.cloud_merge_distance.setToolTip("Keep one existing surface point within this distance. Zero disables merging; chains of points do not collapse into one.")
        for label, widget in (("Quiet-area density", self.cloud_quiet_density),
                ("Loud-area density", self.cloud_loud_density),
                ("Point energy threshold", self.cloud_energy_threshold),
                ("Merging distance", self.cloud_merge_distance)):
            form.addRow(label, widget)
        body.addLayout(form)
        self.scatter_cloud = QPushButton("Generate points")
        body.addWidget(self.scatter_cloud)
        self.cloud_points_status = QLabel("Create a metaball and analyze audio to generate points.")
        self.cloud_points_status.setWordWrap(True)
        body.addWidget(self.cloud_points_status)
        body = self._cloud_section(layout, "Circulation")
        self.circulation_table = self._table(["Curve", "Weight / radius (m)"])
        body.addWidget(self.circulation_table)
        hint = QLabel("Remove points within each curve's pipe radius. Default: 2 m. Set 0 to skip a curve. Regenerate meshes after subtraction.")
        hint.setWordWrap(True)
        body.addWidget(hint)
        self.subtract_points = QPushButton("Subtract points")
        self.subtract_points.clicked.connect(self._subtract_circulation_points)
        body.addWidget(self.subtract_points)
        layout.addStretch()
        generate = QWidget()
        self.generate_layout = QVBoxLayout(generate)
        return cloud, generate

    def _sync_cloud_tables(self):
        origins = self.site_model.origins if self.site_model else []
        self.cloud_origins.setRowCount(len(origins))
        for index, (name, _) in enumerate(origins):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.cloud_origins.setItem(index, 0, item)
            settings = self.viewport._origin_settings.get(index, {})
            check = QCheckBox()
            check.setChecked(settings.get("cloud_include", True))
            scale = self._number(settings.get("radius_scale", 1), .01, 20)
            check.toggled.connect(lambda value, i=index: self.viewport._origin_settings.setdefault(i, {}).update(cloud_include=value))
            scale.valueChanged.connect(lambda value, i=index: self.viewport._origin_settings.setdefault(i, {}).update(radius_scale=value))
            self.cloud_origins.setCellWidget(index, 1, check)
            self.cloud_origins.setCellWidget(index, 2, scale)
        curves = [o for o in self.site_model.objects if o["category"] == "Circulation" and "lines" in o] if self.site_model else []
        self.circulation_table.setRowCount(len(curves))
        for row, obj in enumerate(curves):
            item = QTableWidgetItem(obj["name"])
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.circulation_table.setItem(row, 0, item)
            radius = self._number(obj.get("circulation_radius", 2), 0, 1000, " m")
            radius.valueChanged.connect(lambda value, o=obj: o.update(circulation_radius=value))
            self.circulation_table.setCellWidget(row, 1, radius)

    def _replace_filtered_points(self, points):
        self.viewport._cloud_points = points
        self.viewport.clear_reconstructions()
        self.viewport._draw_cloud_points()
        self.viewport._apply_visibility()
        self.viewport.plotter.render()
        self._rebuild_layers()

    def _subtract_circulation_points(self):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before subtracting points.")
            return
        points = self.viewport._cloud_points
        if points is None:
            self.statusBar().showMessage("Generate points first.")
            return
        curves = [(o["vertices"], o["lines"], o.get("circulation_radius", 2))
                  for o in self.site_model.objects
                  if o["category"] == "Circulation" and "lines" in o] if self.site_model else []
        result, count = subtract_curve_tubes(points, curves)
        self._replace_filtered_points(result)
        message = f"Removed {count:,} circulation points; {result.n_points:,} points remain."
        self.cloud_points_status.setText(message)
        self.statusBar().showMessage(message)

    def _edit_origin_position(self):
        item = self._selected_object
        if not self.site_model or not item or "origin_index" not in item:
            return
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task before moving an origin.")
            self._select_viewport_object(item)
            return
        index = item["origin_index"]
        name, old = self.site_model.origins[index]
        point = np.array([box.value() for box in self.origin_xyz])
        if np.allclose(old, point, rtol=0, atol=1e-9):
            return
        self.site_model.origins[index] = (name, point)
        settings = {i: dict(s) for i, s in self.viewport._origin_settings.items()}
        members = self.viewport._cloud_members
        enabled = self.viewport._metaball_enabled
        deleted = self.viewport._deleted_spheres.copy()
        self.viewport.set_site_model(self.site_model)
        self.viewport._origin_settings = settings
        self.viewport._cloud_members = members
        self.viewport._metaball_enabled = enabled
        self.viewport._deleted_spheres = deleted
        self.viewport._envelope_key = None
        self.viewport.clear_reconstructions()
        for identifier in self._scene_objects:
            self._show_scene_object(identifier)
        self.audio_panel.origins.setItemData(index, point.tolist())
        self._rebuild_layers()
        self._select_viewport_object({**item,"point":point,"from_layers":True})
        self._select_origin(index)
        self._refresh_overlay(self.design_state.parameters())
        self.cloud_points_status.setText("Origin moved. Regenerate surface points and meshes.")
