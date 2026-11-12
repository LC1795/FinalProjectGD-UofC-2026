"""Design parameters. Every control here is read when architecture is generated."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from soundform.models.project import DesignParameters, default_parameters


class GeometryPanel(QWidget):
    generate_requested = Signal()
    guides_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox | QSpinBox] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        root = QVBoxLayout(host)
        root.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Design")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(title)
        note = QLabel(
            "The microphone stays at the centre of the occupant zone. "
            "Moving the zone moves that listening point. It does not place sound sources, "
            "and it does not reconstruct the recorded site."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        root.addWidget(note)

        self.mode = QComboBox()
        self.mode.addItem("Protective terrain", "terrain")
        self.mode.addItem("Directional walls", "walls")
        self.mode.addItem("Protective pavilion", "pavilion")
        self.mode.currentIndexChanged.connect(lambda *_args: self._sync_visibility())
        self.mode.currentIndexChanged.connect(lambda *_args: self.guides_changed.emit())
        root.addWidget(self.mode)

        self.shape = QComboBox()
        self.shape.addItem("Circular site", "circle")
        self.shape.addItem("Rectangular site", "rectangle")
        self.shape.currentIndexChanged.connect(lambda *_args: self._sync_visibility())
        self.shape.currentIndexChanged.connect(lambda *_args: self.guides_changed.emit())

        site = QGroupBox("Site")
        site_form = QFormLayout(site)
        site_form.addRow("Boundary", self.shape)
        self._add(site_form, "site_radius", "Radius (m)", 8, 200, 30, 1, "Radius of a circular site.")
        self._add(site_form, "site_width", "Width (m)", 8, 300, 60, 1, "East-west size of a rectangular site.")
        self._add(site_form, "site_depth", "Depth (m)", 8, 300, 60, 1, "North-south size. +Y is design forward.")
        root.addWidget(site)

        zone = QGroupBox("Occupant zone")
        zone_form = QFormLayout(zone)
        self._add(zone_form, "protected_radius", "Radius (m)", 1, 80, 5, 0.5, "Clear area around the listening point.")
        self._add(zone_form, "protected_center_x", "Centre X (m)", -200, 200, 0, 0.5, "Moves the listening point. Right is +X.")
        self._add(zone_form, "protected_center_y", "Centre Y (m)", -200, 200, 0, 0.5, "Forward is +Y.")
        root.addWidget(zone)

        form_group = QGroupBox("Response")
        form = QFormLayout(form_group)
        self._add(form, "max_height", "Maximum height (m)", 0.2, 80, 8, 0.5, "Upper height of terrain, walls, or the pavilion.")
        self._add(form, "min_height", "Minimum height (m)", 0, 40, 0, 0.1, "Floor or sill. The occupant zone is held to this height in terrain mode.")
        self._add(form, "response_intensity", "Response intensity", 0, 3, 1, 0.05, "Scales how strongly directional score becomes height. Values above 1 saturate the loudest directions.")
        self._add(form, "smoothing", "Form smoothing", 0, 1, 0.35, 0.05, "Smooths the architectural surface or the directional score used to shape it.")
        self._add(form, "rotation_offset_deg", "Rotation offset (°)", -180, 180, 0, 5, "Rotates the decoded field counter-clockwise about up. Zero keeps microphone forward on design +Y.")
        self._add(form, "weight_broadband", "Broadband weight", 0, 2, 0.30, 0.05, "Share of relative broadband energy in the score.")
        self._add(form, "weight_low", "Low-band weight", 0, 2, 0.15, 0.05, "Low-frequency energy is not assumed to matter most.")
        self._add(form, "weight_mid", "Mid-band weight", 0, 2, 0.20, 0.05, "Mid-band energy, 250–2000 Hz with the default limits.")
        self._add(form, "weight_high", "High-band weight", 0, 2, 0.10, 0.05, "High-band energy.")
        self._add(form, "weight_persistence", "Persistence weight", 0, 2, 0.15, 0.05, "Favours directions that stay energetic across the analysed interval.")
        self._add(form, "weight_transient", "Transient weight", 0, 2, 0.05, 0.05, "Favours directions that spike and then drop.")
        self._add(form, "weight_confidence", "Confidence weight", 0, 2, 0.05, 0.05, "Reduces the influence of diffuse or out-of-phase tiles.")
        root.addWidget(form_group)

        self.terrain_group = QGroupBox("Terrain")
        terrain_form = QFormLayout(self.terrain_group)
        resolution = QSpinBox()
        resolution.setRange(16, 140)
        resolution.setValue(64)
        resolution.setToolTip("Samples along each side of the terrain grid.")
        resolution.valueChanged.connect(lambda *_args: self.guides_changed.emit())
        self._spins["grid_resolution"] = resolution
        terrain_form.addRow("Grid resolution", resolution)
        root.addWidget(self.terrain_group)

        self.wall_group = QGroupBox("Walls")
        wall_form = QFormLayout(self.wall_group)
        self._add(wall_form, "wall_thickness", "Thickness (m)", 0.08, 5, 0.45, 0.05, "Radial thickness of a barrier segment.")
        self._add(wall_form, "barrier_spacing", "Barrier spacing (m)", 0.2, 40, 3, 0.5, "Standoff from the occupant-zone edge to the inner face of the walls.")
        self._add(wall_form, "curvature", "Curvature", 0, 1, 0.65, 0.05, "0 is a straight segment. 1 follows the arc.")
        self._add(wall_form, "openness", "Openness", 0, 1, 0.35, 0.05, "Fraction of the quieter directions left without a wall.")
        self._add(wall_form, "entry_width_deg", "Entry width (°)", 5, 120, 28, 1, "Guaranteed opening centred on the quietest direction.")
        root.addWidget(self.wall_group)

        self.pavilion_group = QGroupBox("Pavilion")
        pavilion_form = QFormLayout(self.pavilion_group)
        self._add(pavilion_form, "pavilion_diameter", "Diameter (m)", 6, 80, 18, 0.5, "Outer footprint. Walls stay outside the occupant zone.")
        self._add(pavilion_form, "roof_curvature", "Roof curvature", 0, 1, 0.55, 0.05, "0 slopes toward the highest wall. 1 raises a crown.")
        self._add(pavilion_form, "enclosure", "Enclosure", 0, 1, 0.55, 0.05, "Higher values keep a baseline wall even in quieter directions.")
        self._add(pavilion_form, "opening_width_deg", "Opening width (°)", 5, 150, 32, 1, "Walkable gap toward the quietest direction, under a lintel.")
        root.addWidget(self.pavilion_group)

        self.generate_button = QPushButton("Generate Architecture")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(lambda *_args: self.generate_requested.emit())
        regenerate = QPushButton("Regenerate")
        regenerate.clicked.connect(lambda *_args: self.generate_requested.emit())
        reset = QPushButton("Reset parameters")
        reset.clicked.connect(lambda *_args: self.reset_parameters())
        root.addWidget(self.generate_button)
        root.addWidget(regenerate)
        root.addWidget(reset)
        heuristic = QLabel(
            "Wall and pavilion coverage is a line-of-sight heuristic. "
            "It is not transmission loss and it is not a decibel prediction."
        )
        heuristic.setObjectName("hint")
        heuristic.setWordWrap(True)
        root.addWidget(heuristic)
        root.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll)
        self._sync_visibility()

    def parameters(self) -> DesignParameters:
        data = default_parameters().to_dict()
        data["mode"] = self.mode.currentData()
        data["site_shape"] = self.shape.currentData()
        for key, widget in self._spins.items():
            data[key] = int(widget.value()) if key == "grid_resolution" else float(widget.value())
        return DesignParameters.from_dict(data)

    def set_parameters(self, params: DesignParameters) -> None:
        self.mode.blockSignals(True)
        self.shape.blockSignals(True)
        mode_index = self.mode.findData(params.mode)
        shape_index = self.shape.findData(params.site_shape)
        if mode_index >= 0:
            self.mode.setCurrentIndex(mode_index)
        if shape_index >= 0:
            self.shape.setCurrentIndex(shape_index)
        for key, widget in self._spins.items():
            widget.blockSignals(True)
            widget.setValue(getattr(params, key))
            widget.blockSignals(False)
        self.mode.blockSignals(False)
        self.shape.blockSignals(False)
        self._sync_visibility()

    def set_busy(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy)

    def reset_parameters(self) -> None:
        self.set_parameters(default_parameters())
        self.guides_changed.emit()

    def _add(self, form, key, label, low, high, value, step, tip) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(2 if step < 1 else 1)
        spin.setToolTip(tip)
        spin.valueChanged.connect(lambda *_args: self.guides_changed.emit())
        self._spins[key] = spin
        form.addRow(label, spin)

    def _sync_visibility(self) -> None:
        mode = self.mode.currentData()
        circle = self.shape.currentData() == "circle"
        self._spins["site_radius"].setEnabled(circle)
        self._spins["site_width"].setEnabled(not circle)
        self._spins["site_depth"].setEnabled(not circle)
        self.terrain_group.setVisible(mode == "terrain")
        self.wall_group.setVisible(mode == "walls")
        self.pavilion_group.setVisible(mode == "pavilion")
        # Thickness is shared by walls and the pavilion, and lives on the wall form.
        # Show it with walls; the pavilion form does not duplicate the spin.
        if mode == "pavilion":
            self.wall_group.setVisible(True)
            self.wall_group.setTitle("Wall thickness")
            for key in ("barrier_spacing", "curvature", "openness", "entry_width_deg"):
                self._spins[key].setVisible(False)
            self._label_for("barrier_spacing").setVisible(False)
            self._label_for("curvature").setVisible(False)
            self._label_for("openness").setVisible(False)
            self._label_for("entry_width_deg").setVisible(False)
            self._spins["wall_thickness"].setVisible(True)
            self._label_for("wall_thickness").setVisible(True)
        else:
            self.wall_group.setTitle("Walls")
            for key in ("wall_thickness", "barrier_spacing", "curvature", "openness", "entry_width_deg"):
                visible = mode == "walls"
                self._spins[key].setVisible(visible)
                self._label_for(key).setVisible(visible)
            if mode == "walls":
                self.wall_group.setVisible(True)

    def _label_for(self, key: str):
        widget = self._spins[key]
        form = widget.parentWidget().layout()
        label = form.labelForField(widget) if form is not None else None
        if label is None:
            return widget
        return label
