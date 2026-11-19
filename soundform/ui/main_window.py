"""Soundform main window. Analysis, form generation, and file IO stay out of the widgets."""

from __future__ import annotations

from PySide6.QtCore import QSettings, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QCheckBox,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QTreeWidget,
    QToolButton,
    QSpinBox,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QAbstractItemView,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
)

from soundform.acoustics.ambisonic_decoder import decode_foa
from soundform.acoustics.protection_estimator import estimate_protection
from soundform.acoustics.sound_field import analyze_file
from soundform.audio.loader import load_audio
from soundform.audio.playback import MonitorPlayer, average_channel_downmix, stereo_monitor_downmix
from soundform.audio.site_soundscape import write_site_soundscape
from soundform.errors import AudioLoadError, ExportError, SoundformError
from soundform.export.mesh_export import export_mesh
from soundform.export.project_io import (
    analysis_rows,
    export_analysis_csv,
    export_analysis_json,
    load_project,
    save_project,
)
from soundform.geometry.generate import generate_architecture
from soundform.geometry.pavilion_edit import edit_mesh, selected_mesh
from soundform.geometry.site_model import load_site
from soundform.geometry.cloud_reconstruction import reconstruct_cloud
from soundform.geometry.analysis_envelope import clip_to_terrain
from soundform.geometry.proposal import evaluate_proposal, screening_percent
from soundform.models.project import AnalysisSettings, default_parameters
from soundform.ui.hex_layout import HexLayout
from soundform.ui.analysis_panel import AnalysisPanel
from soundform.ui.audio_panel import AudioPanel
from soundform.ui.geometry_controls import GeometryPanel
from soundform.ui.viewport import Viewport
from soundform.ui.pavilion_editor import PavilionEditor
from soundform.ui.proposal_panel import ProposalPanel


class _Worker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.succeeded.emit(self._fn())
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Soundform")
        self.resize(1440, 900)
        self.audio = None
        self.site_model = None
        self.field = None
        self.mesh = None
        self.report = None
        self._signature = None
        self._pending_signature = None
        self._job = ""
        self._worker: _Worker | None = None
        self._closing = False
        self._shown_frame = -1
        self.player = MonitorPlayer()

        self.audio_panel = AudioPanel()
        self.viewport = Viewport()
        self.geometry_panel = GeometryPanel()
        self.pavilion_editor = PavilionEditor()
        self.pavilion_editor.show_mesh(None)
        self._mesh_undo = []
        self.proposal_panel = ProposalPanel()
        self._proposal_timer = QTimer(self)
        self._proposal_timer.setSingleShot(True)
        self._proposal_timer.setInterval(280)
        self._proposal_timer.timeout.connect(self._edit_proposal)
        self.analysis_panel = AnalysisPanel()

        self.properties_text = QLabel("Click an object in the viewport to inspect it.")
        self.properties_text.setWordWrap(True)
        self.properties_text.setMinimumHeight(140)
        self.properties_text.setTextFormat(Qt.PlainText)
        self.pavilion_editor.layout().insertWidget(0, self.properties_text)
        self._selected_object = None
        properties_host = QWidget()
        properties_layout = QVBoxLayout(properties_host)
        self.metaball_box = QCheckBox("Merge analysis spheres (metaball)")
        self.metaball_box.setChecked(False)
        self.metaball_box.setEnabled(False)
        self.metaball_box.setToolTip("Turn off to inspect separate analysis spheres at the imported origins.")
        properties_layout.addWidget(self.metaball_box)
        self.surface_stats = QLabel("Analyze audio to sample the analysis surface.")
        self.surface_stats.setWordWrap(True)
        self.surface_stats.setTextFormat(Qt.PlainText)
        properties_layout.addWidget(self.surface_stats)
        properties_layout.addWidget(self.pavilion_editor, 1)
        self._selected_origins = set()
        self._syncing_layers = False
        self.origin_form_host = QWidget()
        origin_form = QFormLayout(self.origin_form_host)
        self.origin_radius = QDoubleSpinBox()
        self.origin_radius.setRange(0.5, 150)
        self.origin_radius.setSuffix(" m")
        self.origin_response = QDoubleSpinBox()
        self.origin_response.setRange(0, 5)
        self.origin_response.setSingleStep(0.1)
        origin_form.addRow("Sphere radius", self.origin_radius)
        origin_form.addRow("Design response weight", self.origin_response)
        self.origin_form_host.hide()
        properties_layout.insertWidget(1, self.origin_form_host)
        cloud = QWidget()
        cloud_layout = QVBoxLayout(cloud)
        self.cloud_selection = QLabel("Select pavilion origins in Layers. Ctrl-click to select several.")
        self.cloud_selection.setWordWrap(True)
        cloud_layout.addWidget(self.cloud_selection)
        self.create_cloud = QPushButton("Create metaball from selected origins")
        cloud_layout.addWidget(self.create_cloud)
        cloud_form = QFormLayout()
        self.cloud_strength = QDoubleSpinBox()
        self.cloud_strength.setRange(0.1, 3)
        self.cloud_strength.setValue(1)
        self.cloud_strength.setSingleStep(0.1)
        self.cloud_strength.setToolTip("Higher strength thickens bridges and joins spheres farther apart.")
        cloud_form.addRow("Connection strength", self.cloud_strength)
        self.cloud_radius_scale = QDoubleSpinBox()
        self.cloud_radius_scale.setRange(0.25, 4)
        self.cloud_radius_scale.setValue(1)
        self.cloud_radius_scale.setSingleStep(0.1)
        cloud_form.addRow("Radius scale", self.cloud_radius_scale)
        self.cloud_opacity = QDoubleSpinBox()
        self.cloud_opacity.setRange(0.05, 1)
        self.cloud_opacity.setValue(0.28)
        self.cloud_opacity.setSingleStep(0.05)
        cloud_form.addRow("Opacity", self.cloud_opacity)
        self.cloud_quiet_density = QDoubleSpinBox()
        self.cloud_quiet_density.setRange(0, 100)
        self.cloud_quiet_density.setValue(0.2)
        self.cloud_quiet_density.setSuffix(" points/m²")
        self.cloud_loud_density = QDoubleSpinBox()
        self.cloud_loud_density.setRange(0, 100)
        self.cloud_loud_density.setValue(4)
        self.cloud_loud_density.setSuffix(" points/m²")
        cloud_form.addRow("Quiet-area density", self.cloud_quiet_density)
        cloud_form.addRow("Loud-area density", self.cloud_loud_density)
        self.cloud_energy_threshold = QDoubleSpinBox()
        self.cloud_energy_threshold.setRange(0, 1)
        self.cloud_energy_threshold.setDecimals(3)
        self.cloud_energy_threshold.setSingleStep(0.05)
        self.cloud_energy_threshold.setToolTip("Generate points only where sampled relative energy reaches this value.")
        cloud_form.addRow("Point energy threshold", self.cloud_energy_threshold)
        cloud_layout.addLayout(cloud_form)
        self.scatter_cloud = QPushButton("Generate sound-weighted surface points")
        cloud_layout.addWidget(self.scatter_cloud)
        self.cloud_points_status = QLabel("Create a metaball and analyze audio to generate points.")
        self.cloud_points_status.setWordWrap(True)
        cloud_layout.addWidget(self.cloud_points_status)
        for title, method in (("Delaunay mesh", "delaunay"), ("Poisson reconstruction", "poisson"), ("Voronoi mesh", "voronoi")):
            toggle = QToolButton()
            toggle.setText(title)
            toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            toggle.setCheckable(True)
            toggle.setArrowType(Qt.RightArrow)
            cloud_layout.addWidget(toggle)
            body = QWidget()
            form = QFormLayout(body)
            if method == "delaunay":
                self.delaunay_alpha = QDoubleSpinBox()
                self.delaunay_alpha.setRange(0, 500)
                self.delaunay_alpha.setSingleStep(0.5)
                self.delaunay_alpha.setSuffix(" m")
                self.delaunay_alpha.setToolTip("Zero produces the convex hull. Positive alpha restricts tetrahedron size.")
                form.addRow("Alpha radius (0 = hull)", self.delaunay_alpha)
            elif method == "poisson":
                self.poisson_resolution = QSpinBox()
                self.poisson_resolution.setRange(32, 128)
                self.poisson_resolution.setValue(64)
                self.poisson_resolution.setSingleStep(16)
                form.addRow("Grid resolution", self.poisson_resolution)
            elif method == "voronoi":
                self.voronoi_cells = QSpinBox()
                self.voronoi_cells.setRange(0, 5000)
                self.voronoi_cells.setValue(250)
                self.voronoi_cells.setSpecialValueText("All points")
                self.voronoi_cells.setToolTip("Maximum number of existing cloud points used as cell centers. Subsampling preserves the sound-weighted point distribution.")
                form.addRow("Cell count limit", self.voronoi_cells)
                self.voronoi_gap = QDoubleSpinBox()
                self.voronoi_gap.setRange(0, 5)
                self.voronoi_gap.setDecimals(3)
                self.voronoi_gap.setSingleStep(0.01)
                self.voronoi_gap.setValue(0.08)
                self.voronoi_gap.setSuffix(" m")
                self.voronoi_gap.setToolTip("Approximate surface gap between neighboring cells. Zero creates a continuous tiled surface.")
                form.addRow("Cell gap", self.voronoi_gap)
                self.voronoi_seed = QSpinBox()
                self.voronoi_seed.setRange(0, 999999)
                self.voronoi_seed.setValue(42)
                self.voronoi_seed.setToolTip("Changes which cloud points are used when the cell limit is below the available point count.")
                form.addRow("Pattern seed", self.voronoi_seed)
                self.voronoi_colors = QCheckBox("Color cells individually")
                self.voronoi_colors.setChecked(True)
                form.addRow(self.voronoi_colors)
                note = QLabel("Cells follow the metaball surface using nearest-point distance in 3D. Regenerate after changing these settings.")
                note.setWordWrap(True)
                form.addRow(note)
            button = QPushButton("Generate " + title.lower())
            button.clicked.connect(lambda checked=False, kind=method: self._reconstruct_cloud(kind))
            form.addRow(button)
            body.hide()
            toggle.toggled.connect(
                lambda opened, content=body, tab=toggle: (
                    content.setVisible(opened), tab.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)))
            cloud_layout.addWidget(body)
        cloud_layout.addStretch()
        self.layers_tree = QTreeWidget()
        self.layers_tree.setHeaderLabels(["Layers and objects"])
        self.layers_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.hex_layout = HexLayout(self.viewport, [
            ("AI pavilion", self.proposal_panel),
            ("Properties", properties_host),
            ("View", self.viewport.view_controls),
            ("Analysis", self.analysis_panel),
            ("Audio", self.audio_panel),
            ("Manual", self.geometry_panel),
            ("Cloud", cloud),
            ("Layers", self.layers_tree),
        ])
        self.setCentralWidget(self.hex_layout)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedWidth(120)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Import an FOA WAV, or create a synthetic example.")

        self._wire()
        self._menus()
        if QSettings("Soundform", "Soundform").value("panels_minimized", False, type=bool):
            QTimer.singleShot(0, lambda: self._collapse_panels(True))
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _wire(self) -> None:
        self.audio_panel.site_requested.connect(self._import_site)
        self.audio_panel.origin_changed.connect(self._select_origin)
        self.viewport.object_selected.connect(self._select_viewport_object)
        self.metaball_box.toggled.connect(self.viewport.set_metaball_enabled)
        self.create_cloud.clicked.connect(self._create_cloud)
        self.scatter_cloud.clicked.connect(self._scatter_cloud_points)
        self.cloud_strength.valueChanged.connect(self._change_cloud_strength)
        self.cloud_radius_scale.valueChanged.connect(self._change_cloud_strength)
        self.cloud_opacity.valueChanged.connect(self._change_cloud_strength)
        self.layers_tree.itemSelectionChanged.connect(self._layers_selection)
        self.layers_tree.itemChanged.connect(self._layer_visibility_changed)
        self.origin_radius.valueChanged.connect(self._edit_origin_properties)
        self.origin_response.valueChanged.connect(self._edit_origin_properties)
        self.viewport.envelope_updated.connect(self._update_surface_properties)
        self.proposal_panel.threshold.valueChanged.connect(self._update_surface_properties)
        self.audio_panel.open_requested.connect(self._open_audio)
        self.audio_panel.example_requested.connect(self._write_example)
        self.audio_panel.analyze_requested.connect(self._analyze)
        self.audio_panel.convention_changed.connect(lambda _name: self._load_monitor())
        self.audio_panel.playback_toggled.connect(self._toggle_playback)
        self.audio_panel.playback_stopped.connect(self._stop_playback)
        self.audio_panel.seek_requested.connect(self._seek)
        self.audio_panel.display_changed.connect(self._refresh_views)
        self.audio_panel.analysis_inputs_changed.connect(self._mark_analysis_stale)
        self.geometry_panel.generate_requested.connect(self._generate)
        self.pavilion_editor.selection_changed.connect(self._highlight_pavilion)
        self.pavilion_editor.apply_requested.connect(self._edit_pavilion)
        self.pavilion_editor.delete_requested.connect(lambda: self._edit_pavilion(delete=True))
        self.pavilion_editor.undo_requested.connect(self._undo_pavilion)
        self.geometry_panel.guides_changed.connect(self._refresh_guides)
        self.proposal_panel.generate_requested.connect(self._generate_proposal)
        self.proposal_panel.dimensions_changed.connect(lambda: self._proposal_timer.start())

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self._import_site_action = file_menu.addAction("Import Rhino site…", self._import_site)
        self._clear_site_action = file_menu.addAction("Remove Rhino site", self._clear_site)
        self._clear_site_action.setEnabled(False)
        file_menu.addAction("Import WAV…", self.audio_panel._import)
        file_menu.addAction("Create synthetic example…", self.audio_panel._example)
        file_menu.addSeparator()
        file_menu.addAction("Open project…", self._open_project)
        file_menu.addAction("Save project…", self._save_project)
        file_menu.addSeparator()
        file_menu.addAction("Export OBJ…", lambda: self._export_mesh(".obj"))
        file_menu.addAction("Export STL…", lambda: self._export_mesh(".stl"))
        file_menu.addAction("Export analysis JSON…", self._export_json)
        file_menu.addAction("Export analysis CSV…", self._export_csv)
        file_menu.addSeparator()
        file_menu.addAction("Reset project", self._reset_project)
        file_menu.addAction("Quit", self.close)
        view_menu = self.menuBar().addMenu("View")
        for label, name in (("Top", "top"), ("Front", "front"), ("Side", "side"), ("Perspective", "perspective")):
            view_menu.addAction(label, lambda checked=False, view=name: self.viewport.set_view(view))
        view_menu.addAction("Reset view", self.viewport.reset_view)
        view_menu.addSeparator()
        view_menu.addAction("Collapse panels", lambda: self._collapse_panels(True))
        view_menu.addAction("Expand panels", lambda: self._collapse_panels(False))
        minimized = view_menu.addAction("Start panels minimized")
        minimized.setCheckable(True)
        settings = QSettings("Soundform", "Soundform")
        minimized.setChecked(settings.value("panels_minimized", False, type=bool))
        minimized.toggled.connect(lambda value: settings.setValue("panels_minimized", value))

    def _import_site(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Rhino site", "", "Rhino model (*.3dm)")
        if path:
            self._start("site", lambda: load_site(path))

    def _set_site(self, model) -> None:
        self.site_model = model
        self.audio_panel.origins.blockSignals(True)
        self.audio_panel.origins.clear()
        if model is not None:
            for name, point in model.origins:
                self.audio_panel.origins.addItem(name, point.tolist())
        self.audio_panel.origins.setEnabled(model is not None and bool(model.origins))
        self.audio_panel.origins.blockSignals(False)
        self.viewport.set_site_model(model)
        self._selected_origins.clear()
        self.metaball_box.blockSignals(True)
        self.metaball_box.setChecked(False)
        self.metaball_box.setEnabled(False)
        self.metaball_box.blockSignals(False)
        self._rebuild_layers()
        self._select_viewport_object(None)
        self._clear_site_action.setEnabled(model is not None)
        self._refresh_overlay(self.geometry_panel.parameters())

    def _collapse_panels(self, collapsed):
        self.hex_layout.set_collapsed(collapsed)

    def _select_origin(self, index):
        if self.site_model is None or not 0 <= index < len(self.site_model.origins):
            return
        name, point = self.site_model.origins[index]
        params = self.geometry_panel.parameters()
        params.protected_center_x, params.protected_center_y = map(float, point[:2])
        self.geometry_panel.set_parameters(params)
        self._refresh_guides()
        self.statusBar().showMessage(f"Active pavilion origin: {name}. Generate to build at this point.")

    def _rebuild_layers(self):
        self._syncing_layers = True
        self.layers_tree.clear()
        groups = {}
        def add(layer, label, data):
            if layer not in groups:
                groups[layer] = QTreeWidgetItem(self.layers_tree, [layer])
                groups[layer].setData(0, Qt.UserRole + 1, layer)
                groups[layer].setFlags(groups[layer].flags() | Qt.ItemIsUserCheckable)
                groups[layer].setCheckState(
                    0, Qt.Checked if self.viewport._layer_visibility.get(layer, True) else Qt.Unchecked)
            item = QTreeWidgetItem(groups[layer], [label])
            item.setData(0, Qt.UserRole, data)
            key = self.viewport.object_visibility_key(data)
            item.setData(0, Qt.UserRole + 2, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if self.viewport._object_visibility.get(key, True) else Qt.Unchecked)
            if data.get("category") == "Pavilion Origin" and data.get("origin_index") in self._selected_origins:
                item.setSelected(True)
            return item
        if self.site_model is not None:
            for obj in self.site_model.objects:
                add(obj["layer"], obj["name"], obj)
            for i, (name, point) in enumerate(self.site_model.origins):
                data = {"name": name, "category": "Pavilion Origin", "point": point, "origin_index": i}
                add("Pavilion Origin", name, data)
                add("Analysis spheres", f"{name} — sphere",
                    {**data, "category": "Analysis sphere"})
            for name, point in self.site_model.sources:
                add("Sound Sources", name, {"name": name, "category": "Sound Sources", "point": point})
        if self.mesh is not None:
            from soundform.geometry.pavilion_edit import part_labels
            labels = part_labels(self.mesh)
            for label in sorted(set(labels)):
                cell = next(i for i, value in enumerate(labels) if value == label)
                add("Pavilion", label, {"name": label, "category": "Pavilion", "cell": cell})
        if self.viewport._cloud_members:
            add("Cloud", "Metaball", {"name": "Metaball", "category": "Analysis envelope"})
        if self.viewport._cloud_points is not None:
            add("Cloud", "Surface points", {"name": "Surface points", "category": "Surface points"})
        for method, mesh in self.viewport._reconstructions.items():
            label = method.title() + " mesh"
            add(label, label, {"name": label, "category": "Reconstruction",
                              "vertices": mesh.points, "faces": mesh.faces.reshape(-1, 4)[:, 1:]})
        self.layers_tree.collapseAll()
        self._syncing_layers = False

    def _layer_visibility_changed(self, item, column):
        if self._syncing_layers:
            return
        layer = item.data(0, Qt.UserRole + 1)
        if layer is not None:
            self.viewport.set_layer_visible(layer, item.checkState(0) == Qt.Checked)
        else:
            key = item.data(0, Qt.UserRole + 2)
            if key is not None:
                self.viewport.set_object_visible(key, item.checkState(0) == Qt.Checked)

    def _reconstruct_cloud(self, method):
        points = self.viewport._cloud_points
        if points is None or points.n_points < 12:
            QMessageBox.information(self, "Cloud reconstruction", "Generate at least 12 cloud points first.")
            return
        source_id = id(points)
        snapshot = points.copy()
        terrain = None if self.viewport._terrain is None else self.viewport._terrain.copy()
        boundary = None if self.viewport._cloud_surface is None else self.viewport._cloud_surface.copy()
        alpha = self.delaunay_alpha.value()
        resolution = self.poisson_resolution.value()
        voronoi_options = dict(voronoi_cells=self.voronoi_cells.value(),
                               voronoi_gap=self.voronoi_gap.value(), voronoi_seed=self.voronoi_seed.value())

        def job():
            mesh = reconstruct_cloud(snapshot, method, alpha, resolution, boundary=boundary, **voronoi_options)
            return method, clip_to_terrain(mesh, terrain).triangulate(), source_id

        self._start("cloud-reconstruction", job)

    def _layers_selection(self):
        if self._syncing_layers:
            return
        items = [item.data(0, Qt.UserRole) for item in self.layers_tree.selectedItems()
                 if item.data(0, Qt.UserRole) is not None]
        self._selected_origins = {item["origin_index"] for item in items if "origin_index" in item}
        self.cloud_selection.setText(f"{len(self._selected_origins)} pavilion origin(s) selected.")
        self.viewport.set_origin_selection(self._selected_origins)
        if items:
            self._select_viewport_object({**items[-1], "from_layers": True})

    def _edit_origin_properties(self):
        if self._selected_object is None or "origin_index" not in self._selected_object:
            return
        indices = self._selected_origins or {self._selected_object["origin_index"]}
        for index in indices:
            self.viewport._origin_settings[index] = {
                "radius": self.origin_radius.value(),
                "response_weight": self.origin_response.value(),
            }
        self._refresh_overlay(self.geometry_panel.parameters())

    def _change_cloud_strength(self, value):
        self.viewport._cloud_strength = self.cloud_strength.value()
        self.viewport._cloud_radius_scale = self.cloud_radius_scale.value()
        self.viewport._cloud_opacity = self.cloud_opacity.value()
        if self.viewport._cloud_members:
            self._refresh_overlay(self.geometry_panel.parameters())

    def _scatter_cloud_points(self):
        try:
            count, capped = self.viewport.generate_cloud_points(
                self.cloud_quiet_density.value(), self.cloud_loud_density.value(),
                self.cloud_energy_threshold.value())
        except ValueError as error:
            QMessageBox.information(self, "Surface points", str(error))
            return
        message = f"{count:,} surface points generated."
        if capped:
            message += " Limited to 50,000 points. Reduce the densities for a smaller array."
        self.cloud_points_status.setText(message)
        self._rebuild_layers()
        self.statusBar().showMessage(message)

    def _create_cloud(self):
        if not self._selected_origins:
            self.statusBar().showMessage("Select pavilion origin points or their spheres in Layers first.")
            return
        self.viewport._cloud_members = tuple(sorted(self._selected_origins))
        self.viewport._cloud_strength = self.cloud_strength.value()
        self.metaball_box.setEnabled(True)
        self.metaball_box.setChecked(True)
        self.viewport.set_metaball_enabled(True)
        self._rebuild_layers()
        self._refresh_overlay(self.geometry_panel.parameters())
        self.statusBar().showMessage(f"Metaball created from {len(self._selected_origins)} selected origin(s).")

    def _update_surface_properties(self):
        data = self.viewport.surface_analysis(self.proposal_panel.threshold.value())
        if data is None:
            self.surface_stats.setText("Analyze audio to sample the analysis surface.")
            return
        self.surface_stats.setText(
            f'Surface area: {data["area_m2"]:.1f} m²\n'
            f'Mean relative energy: {data["mean_relative_energy"]:.3f}\n'
            f'Area above threshold: {data["above_threshold_area_percent"]:.1f}%\n'
            "Each origin uses the loaded recording."
        )

    def _select_viewport_object(self, item):
        self._selected_object = item
        self.origin_form_host.hide()
        if item is not None and "origin_index" in item:
            index = item["origin_index"]
            if not item.get("from_layers"):
                if item.get("extend_selection"):
                    self._selected_origins.symmetric_difference_update({index})
                else:
                    self._selected_origins = {index}
                self._syncing_layers = True
                iterator = QTreeWidgetItemIterator(self.layers_tree)
                while iterator.value():
                    node = iterator.value()
                    data = node.data(0, Qt.UserRole)
                    node.setSelected(bool(data and data.get("origin_index") in self._selected_origins
                                          and data.get("category") == "Pavilion Origin"))
                    iterator += 1
                self._syncing_layers = False
            settings = self.viewport._origin_settings.get(index, {})
            self.origin_radius.blockSignals(True)
            self.origin_response.blockSignals(True)
            self.origin_radius.setValue(settings.get("radius", self.geometry_panel.parameters().pavilion_diameter / 2))
            self.origin_response.setValue(settings.get("response_weight", 1))
            self.origin_radius.blockSignals(False)
            self.origin_response.blockSignals(False)
            self.origin_form_host.show()
            self.viewport.set_origin_selection(self._selected_origins)
            self.cloud_selection.setText(f"{len(self._selected_origins)} pavilion origin(s) selected.")
            if item["category"] == "Analysis sphere":
                point = self.site_model.origins[index][1]
                item = {**item, "point": point}
        if item is not None:
            self.hex_layout.set_open("Properties", True)
        if item is None:
            self.properties_text.setText("Click an object in the viewport to inspect it.")
            self.pavilion_editor.setEnabled(False)
            self.viewport.set_pavilion_selection(None)
            return
        self.pavilion_editor.setEnabled(True)
        if item["category"] in ("Analysis envelope", "Surface points"):
            self.pavilion_editor.show_mesh(None)
            self.properties_text.setText(item["category"])
            self.viewport.set_pavilion_selection(None)
            dataset = self.viewport._cloud_points if item["category"] == "Surface points" else self.viewport._envelope
            self.viewport._focus_points = dataset.points
            self._update_surface_properties()
            return
        if item["category"] == "Pavilion" and self.mesh is not None:
            from soundform.geometry.pavilion_edit import part_labels
            cell = item["cell"]
            labels = part_labels(self.mesh)
            if not 0 <= cell < len(labels):
                return
            label = labels[cell]
            self.pavilion_editor.show_mesh(self.mesh, bool(self._mesh_undo))
            index = self.pavilion_editor.selection.findData(label)
            self.pavilion_editor.selection.setCurrentIndex(max(0, index))
            part = selected_mesh(self.mesh, label if index >= 0 else None)
            dimensions = ", ".join(f"{v:.3f}" for v in part.extents)
            self.properties_text.setText(
                f"Pavilion component\n{label}\nSize XYZ (m): {dimensions}\nTriangles: {len(part.faces)}"
            )
            self._highlight_pavilion()
        else:
            self.pavilion_editor.show_mesh(None)
            text = f'{item["name"]}\n{item["category"]}'
            if "vertices" in item:
                vertices = item["vertices"]
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                text += (f'\nLayer: {item["layer"]}\nTriangles: {len(item["faces"])}'
                         f'\nSize XYZ (m): {", ".join(f"{v:.3f}" for v in high - low)}'
                         f'\nBounds min (m): {", ".join(f"{v:.3f}" for v in low)}'
                         f'\nBounds max (m): {", ".join(f"{v:.3f}" for v in high)}')
                from soundform.geometry.mesh_utils import as_mesh
                self.viewport.set_pavilion_selection(as_mesh(vertices, item["faces"]))
            else:
                text += "\nPosition XYZ (m): " + ", ".join(f"{v:.3f}" for v in item["point"])
                if "origin_index" in item:
                    self.audio_panel.origins.setCurrentIndex(item["origin_index"])
                self.viewport.set_pavilion_selection(None)
                import numpy as np
                self.viewport._focus_points = np.array([item["point"]]) + np.array([
                    [-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]
                ])
            self.properties_text.setText(text)

    def _clear_site(self) -> None:
        self._set_site(None)
        self.statusBar().showMessage("Rhino site removed.")

    def _open_audio(self, path: str) -> None:
        try:
            audio = load_audio(path)
        except AudioLoadError as exc:
            QMessageBox.warning(self, "Could not import", str(exc))
            return
        self.audio = audio
        self.field = None
        self.mesh = None
        self.report = None
        self._signature = None
        self.player.stop()
        self.audio_panel.show_audio(audio)
        self._load_monitor()
        self._show_editable_mesh(None)
        self.analysis_panel.clear()
        self.analysis_panel.show_messages(audio.detection.messages)
        self._refresh_guides()
        self.statusBar().showMessage(f"Loaded {audio.name}. {audio.channels} channels, {audio.sample_rate} Hz.")

    def _write_example(self, path: str) -> None:
        try:
            info = write_site_soundscape(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not write the example", str(exc))
            return
        self._open_audio(info["path"])
        QMessageBox.information(self, "Synthetic example", info["description"])

    def _load_monitor(self) -> None:
        if self.audio is None:
            self.player.clear()
            return
        convention = self.audio_panel.current_convention()
        try:
            if convention and self.audio.channels in (4, 16):
                stereo = stereo_monitor_downmix(decode_foa(self.audio.samples[:, :4], convention))
                note = "Stereo monitor of the decoded Ambisonic bed. This is not binaural and it is not the analysis."
            else:
                stereo = average_channel_downmix(self.audio.samples)
                note = "Non-spatial channel average. Choose a convention for an Ambisonic stereo monitor."
            self.player.load(stereo, self.audio.sample_rate)
            self.audio_panel.set_monitor_note(note)
        except Exception as exc:
            self.audio_panel.set_monitor_note(f"Playback is unavailable ({exc}). Analysis can still run.")

    def _toggle_playback(self) -> None:
        if self.audio is None:
            return
        try:
            if self.player.playing:
                self.player.pause()
            else:
                self.player.play()
                if self.player.playing and self.field is not None:
                    self.audio_panel.time_mode.setCurrentIndex(self.audio_panel.time_mode.findData("frame"))
                    self._refresh_views()
        except Exception as exc:
            QMessageBox.warning(self, "Playback", str(exc))
        self.audio_panel.set_playing(self.player.playing)

    def _stop_playback(self) -> None:
        self.player.stop()
        self.audio_panel.set_playing(False)
        self.audio_panel.set_position(0.0)
        self._refresh_views()

    def _seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        self._refresh_views()

    def _poll(self) -> None:
        if self.audio is None:
            return
        self.audio_panel.set_position(self.player.position_seconds)
        if self.player.playing != (self.audio_panel.play_button.text() == "Pause"):
            self.audio_panel.set_playing(self.player.playing)
        if self.field is None or self.audio_panel.settings().view_time_mode != "frame":
            return
        index = self.field.frame_index_at(self.player.position_seconds)
        if index != self._shown_frame:
            self._shown_frame = index
            self._refresh_views()

    def _analyze(self) -> None:
        if self.audio is None or self.audio_panel.current_convention() is None:
            QMessageBox.information(self, "Convention required", "Choose an Ambisonic convention before analysis.")
            return
        settings = self.audio_panel.settings()
        if settings.end_s - settings.start_s < 0.25:
            QMessageBox.warning(self, "Interval", "Select an analysis interval of at least 0.25 seconds.")
            return
        samples = self.audio.samples
        sample_rate = self.audio.sample_rate
        duration = self.audio.duration
        convention = self.audio_panel.current_convention()
        limits = settings.band_limits()
        signature = self._signature_for(convention, settings)

        def job():
            return analyze_file(samples, sample_rate, convention, settings.start_s, settings.end_s, duration, limits)

        self._pending_signature = signature
        self._start("analyze", job)

    def _generate(self) -> None:
        if self.field is None:
            QMessageBox.information(
                self,
                "Analyze first",
                "Analyze the recording before generating architecture. Soundform does not use placed sound sources.",
            )
            return
        if self._signature != self._signature_for(self.audio_panel.current_convention(), self.audio_panel.settings()):
            QMessageBox.information(
                self,
                "Analyze again",
                "The interval, frequency limits, or channel convention changed. Analyze Audio again so the form matches those settings.",
            )
            return
        params = self.geometry_panel.parameters()
        field = self.field

        def job():
            mesh, response, used, warnings = generate_architecture(field, params)
            bands = {name: field.aggregate(name) for name in ("broadband", "low", "mid", "high")}
            report = estimate_protection(mesh, response, used, bands)
            return mesh, response, used, warnings, report

        self._start("generate", job)

    def _generate_proposal(self) -> None:
        self._start_proposal()

    def _edit_proposal(self) -> None:
        if self.proposal_panel.proposal is not None:
            self._start_proposal(self.proposal_panel.edited_proposal())

    def _start_proposal(self, definition=None) -> None:
        if self.field is None or self._signature != self._signature_for(
            self.audio_panel.current_convention(), self.audio_panel.settings()
        ):
            QMessageBox.information(self, "Analyze first", "Analyze the current recording before designing a pavilion.")
            return
        self._proposal_timer.stop()
        panel = self.proposal_panel
        editing = definition is not None
        params = self.geometry_panel.parameters()
        field = self.field
        threshold = panel.applied_threshold if editing else panel.threshold.value()
        prompt = panel.prompt.toPlainText().strip()
        sketches = panel.sketch_images()
        surface_context = self.viewport.surface_analysis(threshold)
        if surface_context is not None:
            for sample in surface_context["samples"]:
                sample["xyz_m"][0] -= params.protected_center_x
                sample["xyz_m"][1] -= params.protected_center_y
            surface_context["coordinates"] = "Local pavilion origin at ground level, XYZ in metres."
            surface_context["metaball_enabled"] = self.metaball_box.isChecked()
        if not editing and not (prompt or sketches):
            QMessageBox.information(self, "Add design input", "Enter a design prompt or attach a concept sketch.")
            return

        def job():
            result = evaluate_proposal(field, params, threshold, definition, prompt, sketches, online=True,
                                       surface_context=surface_context)
            return result, threshold, editing

        self._start("proposal", job)

    def _show_proposal(self, result) -> None:
        (definition, mesh, report, baseline, warnings), threshold, editing = result
        panel = self.proposal_panel
        if editing:
            panel.proposal = definition
        else:
            panel.show_proposal(definition)
        panel.applied_threshold = threshold
        self.mesh, self.report = mesh, report
        self._show_editable_mesh(mesh)
        screened = screening_percent(report)

        def formatted(value):
            return f"{value:.1f}%" if value is not None else "unavailable"

        panel.result.setText(
            f"Above-threshold energy screened ({threshold:g} dB relative to peak)\n"
            f"Proposal {formatted(screened)} · Standard pavilion {formatted(baseline)}"
        )
        self.analysis_panel.show_report(report, self.field)
        self.analysis_panel.show_messages([*self.field.warnings, *warnings, *report.warnings])
        self.statusBar().showMessage("Pavilion preview updated. Screening is a line-of-sight estimate.")

    def _show_editable_mesh(self, mesh, *, new=True) -> None:
        if new:
            self._mesh_undo.clear()
        self.viewport.set_mesh(mesh)
        self._rebuild_layers()
        self.pavilion_editor.show_mesh(mesh, bool(self._mesh_undo))
        self._highlight_pavilion()

    def _highlight_pavilion(self) -> None:
        selection = None
        if self.hex_layout.is_open("Properties") and (
            self._selected_object is None or self._selected_object.get("category") == "Pavilion"
        ):
            selection = selected_mesh(self.mesh, self.pavilion_editor.selection.currentData())
        self.viewport.set_pavilion_selection(selection)

    def _edit_pavilion(self, *, delete=False) -> None:
        if self.mesh is None:
            return
        panel = self.pavilion_editor
        offset = tuple(widget.value() for widget in panel.offset)
        rotation, scale = panel.rotation.value(), panel.scale.value()
        if not delete and offset == (0, 0, 0) and rotation == 0 and scale == 1:
            return
        part = panel.selection.currentData()
        self._mesh_undo.append(self.mesh)
        self._mesh_undo = self._mesh_undo[-10:]
        self.mesh = edit_mesh(self.mesh, part, delete=delete, offset=offset, rotation=rotation, scale=scale)
        self._pavilion_changed()

    def _undo_pavilion(self) -> None:
        if self._mesh_undo:
            self.mesh = self._mesh_undo.pop()
            self._pavilion_changed()

    def _pavilion_changed(self) -> None:
        self._proposal_timer.stop()
        self.proposal_panel.show_proposal(None)
        self.report = None
        self._show_editable_mesh(self.mesh, new=False)
        self.analysis_panel.show_report(None, self.field)
        self.statusBar().showMessage("Pavilion edited. The previous screening estimate was cleared.")

    def _start(self, kind: str, fn) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task to finish.")
            return
        self._job = kind
        self._worker = _Worker(fn)
        self._worker.succeeded.connect(self._job_ok)
        self._worker.failed.connect(self._job_fail)
        self._progress.setVisible(True)
        self.audio_panel.set_busy(True)
        self.geometry_panel.set_busy(True)
        self.proposal_panel.setEnabled(False)
        self.pavilion_editor.setEnabled(False)
        self._import_site_action.setEnabled(False)
        self._clear_site_action.setEnabled(False)
        messages = {"analyze": "Analyzing the sound field.", "site": "Importing the Rhino site.",
                    "cloud-reconstruction": "Reconstructing the cloud surface."}
        self.statusBar().showMessage(messages.get(kind, "Generating architecture."))
        self._worker.start()

    def _job_ok(self, result) -> None:
        self._end_busy()
        if self._closing:
            return
        if self._job == "cloud-reconstruction":
            method, mesh, source_id = result
            if id(self.viewport._cloud_points) != source_id:
                self.statusBar().showMessage("Cloud points changed during reconstruction. Generate the mesh again.")
                return
            if not mesh.n_cells:
                self.statusBar().showMessage("The reconstruction lies below the terrain. Adjust the points.")
                return
            self.viewport.show_reconstruction(method, mesh, color_by_cell=method == "voronoi" and self.voronoi_colors.isChecked())
            self._rebuild_layers()
            self.statusBar().showMessage(f"{method.title()} mesh generated.")
            return
        if self._job == "site":
            self._set_site(result)
            params = self.geometry_panel.parameters()
            params.protected_center_x = params.protected_center_y = 0.0
            self.geometry_panel.set_parameters(params)
            self.mesh = self.report = None
            self.proposal_panel.show_proposal(None)
            self._show_editable_mesh(None)
            self._refresh_guides()
            if self.field is not None:
                self._refresh_views()
                self.analysis_panel.show_report(None, self.field)
            self.viewport.set_view("perspective")
            self.viewport.plotter.reset_camera()
            self.statusBar().showMessage(f"Imported {result.path.name}.")
            if result.warnings:
                QMessageBox.information(self, "Rhino site import", "\n".join(result.warnings))
            return
        if self._job == "proposal":
            self._show_proposal(result)
            return
        if self._job == "analyze":
            self.proposal_panel.show_proposal(None)
            self.field = result
            self._signature = self._pending_signature
            self.mesh = None
            self.report = None
            self._show_editable_mesh(None)
            self._shown_frame = -1
            self._refresh_views()
            self.analysis_panel.show_report(None, self.field)
            self.analysis_panel.show_messages(result.warnings)
            self.statusBar().showMessage("Analysis complete. Adjust the design, then generate architecture.")
            return
        mesh, _response, used, warnings, report = result
        self.proposal_panel.show_proposal(None)
        self.mesh = mesh
        self.report = report
        self.geometry_panel.set_parameters(used)
        self._show_editable_mesh(mesh)
        self._refresh_guides()
        notes = []
        if self.field is not None:
            notes.extend(self.field.warnings)
        notes.extend(warnings)
        notes.extend(report.warnings)
        self.analysis_panel.show_report(report, self.field)
        self.analysis_panel.show_messages(notes)
        self.statusBar().showMessage(
            f"Generated {used.mode}. {len(mesh.vertices)} vertices, {len(mesh.faces)} faces. "
            "The protection comparison is a heuristic."
        )

    def _job_fail(self, message: str) -> None:
        self._end_busy()
        if self._closing:
            return
        QMessageBox.warning(self, "Soundform", message)
        self.statusBar().showMessage(message)

    def _end_busy(self) -> None:
        self._progress.setVisible(False)
        self.audio_panel.set_busy(False)
        self.geometry_panel.set_busy(False)
        self.proposal_panel.setEnabled(True)
        self.pavilion_editor.setEnabled(True)
        self._import_site_action.setEnabled(True)
        self._clear_site_action.setEnabled(self.site_model is not None)

    def _refresh_guides(self) -> None:
        params = self.geometry_panel.parameters()
        self.viewport.set_guides(params)
        self._refresh_overlay(params)
        if self.mesh is not None:
            self.statusBar().showMessage("Guides follow the new parameters. Generate Architecture to rebuild the mesh.")

    def _refresh_views(self) -> None:
        if self.field is None:
            return
        settings = self.audio_panel.settings()
        time_s = self.player.position_seconds if settings.view_time_mode == "frame" else None
        self.analysis_panel.show_field(self.field, settings.view_band, settings.view_smoothing, time_s)
        if self.report is not None:
            self.analysis_panel.show_report(self.report, self.field)
        self._refresh_overlay(self.geometry_panel.parameters())

    def _refresh_overlay(self, params) -> None:
        if self.field is None:
            self.viewport.set_overlay(None, None, None, params)
            return
        settings = self.audio_panel.settings()
        time_s = self.player.position_seconds if settings.view_time_mode == "frame" else None
        values = self.field.display_map(settings.view_band, time_s, settings.view_smoothing)
        self.viewport.set_overlay(self.field.azimuth_deg, self.field.elevation_deg, values, params)

    def _mark_analysis_stale(self) -> None:
        if self.field is None:
            return
        if self._signature != self._signature_for(self.audio_panel.current_convention(), self.audio_panel.settings()):
            self.statusBar().showMessage("Analysis settings changed. Press Analyze Audio to apply them.")

    def _signature_for(self, convention: str | None, settings: AnalysisSettings):
        limits = tuple((name, tuple(bounds)) for name, bounds in sorted(settings.band_limits().items()))
        return (convention, round(settings.start_s, 3), round(settings.end_s, 3), limits)

    def _save_project(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Save project", "soundform-project.json", "Soundform project (*.json)")
        if not path:
            return
        recipe = self.proposal_panel.proposal
        if recipe is not None and "recipe_version" not in recipe:
            recipe = None
        try:
            save_project(
                path,
                None if self.audio is None else self.audio.path,
                self.audio_panel.current_convention(),
                self.audio_panel.settings(),
                self.geometry_panel.parameters(),
                self.field,
                self.mesh,
                site_path=str(self.site_model.path) if self.site_model else None,
                recipe=recipe,
                recipe_threshold=self.proposal_panel.applied_threshold if recipe is not None else None,
            )
        except (ExportError, OSError, SoundformError) as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.statusBar().showMessage(f"Saved project {path}. The audio file itself was not embedded.")

    def _open_project(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open project", "", "Soundform project (*.json)")
        if not path:
            return
        try:
            record = load_project(path)
        except ExportError as exc:
            QMessageBox.warning(self, "Could not open", str(exc))
            return
        self.player.stop()
        self.proposal_panel.show_proposal(None)
        self.geometry_panel.set_parameters(record.design)
        self.audio_panel.apply_settings(record.analysis_settings)
        self.field = record.sound_field
        self.mesh = record.mesh
        self.report = None
        self.audio = None
        messages = list(record.warnings)
        site = None
        if record.site_path:
            try:
                site = load_site(record.site_path)
                messages.extend(site.warnings)
            except (ValueError, OSError) as exc:
                messages.append(f"Could not reload the Rhino site: {exc}")
        self._set_site(site)
        if record.audio_found and record.audio_path:
            try:
                self.audio = load_audio(record.audio_path)
                self.audio_panel.show_audio(self.audio)
                self.audio_panel.apply_settings(record.analysis_settings)
                if record.convention:
                    self.audio_panel.set_convention(record.convention)
                self._load_monitor()
                messages = self.audio.detection.messages + messages
            except AudioLoadError as exc:
                messages.append(str(exc))
                self.audio_panel.clear()
        else:
            self.audio_panel.clear()
            if record.convention:
                self.audio_panel.set_convention(record.convention)
        if self.field is not None and record.convention:
            self._signature = self._signature_for(record.convention, record.analysis_settings)
        else:
            self._signature = None
        self._show_editable_mesh(self.mesh)
        if record.recipe is not None:
            self.proposal_panel.show_proposal(record.recipe)
            self.proposal_panel.applied_threshold = record.recipe_threshold
            self.proposal_panel.threshold.setValue(record.recipe_threshold)
        self._refresh_guides()
        if self.field is not None:
            self._refresh_views()
            messages = self.field.warnings + messages
        else:
            self.analysis_panel.clear()
        self.analysis_panel.show_messages(messages)
        self.statusBar().showMessage("Project opened. Generate again if you want a new protection estimate.")

    def _export_mesh(self, suffix: str) -> None:
        if self.mesh is None:
            QMessageBox.information(self, "Nothing to export", "Generate architecture before exporting a mesh.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export mesh",
            f"soundform{suffix}",
            f"{suffix[1:].upper()} (*{suffix})",
        )
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix
        try:
            info = export_mesh(self.mesh, path)
        except ExportError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Mesh exported",
            f"Wrote {info['vertices']} vertices and {info['faces']} faces to\n{info['path']}",
        )

    def _export_json(self) -> None:
        if self.field is None:
            QMessageBox.information(self, "Nothing to export", "Analyze audio before exporting a summary.")
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export analysis JSON", "soundform-analysis.json", "JSON (*.json)")
        if not path:
            return
        coverage = None if self.report is None else self.report.coverage
        rows = analysis_rows(self.field, coverage)
        export_analysis_json(path, self.field, rows)
        self.statusBar().showMessage(f"Wrote analysis JSON to {path}.")

    def _export_csv(self) -> None:
        if self.field is None:
            QMessageBox.information(self, "Nothing to export", "Analyze audio before exporting a summary.")
            return
        path, _filter = QFileDialog.getSaveFileName(self, "Export analysis CSV", "soundform-analysis.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            coverage = None if self.report is None else self.report.coverage
            export_analysis_csv(path, analysis_rows(self.field, coverage))
        except ExportError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Wrote analysis CSV to {path}.")

    def _reset_project(self) -> None:
        answer = QMessageBox.question(self, "Reset project", "Clear the Rhino site, recording, analysis, mesh, and design parameters?")
        if answer != QMessageBox.Yes:
            return
        self.player.stop()
        self.audio = None
        self.field = None
        self.mesh = None
        self.report = None
        self._signature = None
        self.audio_panel.clear()
        self.proposal_panel.show_proposal(None)
        self.geometry_panel.set_parameters(default_parameters())
        self._show_editable_mesh(None)
        self.analysis_panel.clear()
        self._set_site(None)
        self._refresh_guides()
        self.statusBar().showMessage("Project reset.")

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Wait for the current task to finish before closing.")
            event.ignore()
            return
        self._closing = True
        self.player.stop()
        self.viewport.shutdown()
        super().closeEvent(event)
