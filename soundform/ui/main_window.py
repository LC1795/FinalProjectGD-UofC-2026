"""Soundform main window. Analysis, form generation, and file IO stay out of the widgets."""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
)

from soundform.acoustics.ambisonic_decoder import decode_foa
from soundform.acoustics.protection_estimator import estimate_protection
from soundform.acoustics.sound_field import analyze_file
from soundform.audio.loader import load_audio
from soundform.audio.playback import MonitorPlayer, average_channel_downmix, stereo_monitor_downmix
from soundform.audio.synthetic import write_synthetic_example
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
from soundform.models.project import AnalysisSettings, default_parameters
from soundform.ui.analysis_panel import AnalysisPanel
from soundform.ui.audio_panel import AudioPanel
from soundform.ui.geometry_controls import GeometryPanel
from soundform.ui.viewport import Viewport


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
        self.analysis_panel = AnalysisPanel()

        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(self.audio_panel)
        left.setMinimumWidth(390)
        center = QSplitter(Qt.Vertical)
        center.addWidget(self.viewport)
        center.addWidget(self.analysis_panel)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(self.geometry_panel)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 700, 340])
        self.geometry_panel.setMinimumWidth(300)
        self.setCentralWidget(splitter)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedWidth(120)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Import an FOA WAV, or create a synthetic example.")

        self._wire()
        self._menus()
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _wire(self) -> None:
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
        self.geometry_panel.guides_changed.connect(self._refresh_guides)

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Import WAV…", self.audio_panel.import_button.click)
        file_menu.addAction("Create synthetic example…", self.audio_panel.example_button.click)
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
        self.viewport.set_mesh(None)
        self.analysis_panel.clear()
        self.analysis_panel.show_messages(audio.detection.messages)
        self._refresh_guides()
        self.statusBar().showMessage(f"Loaded {audio.name}. {audio.channels} channels, {audio.sample_rate} Hz.")

    def _write_example(self, path: str) -> None:
        try:
            info = write_synthetic_example(path)
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
            if convention and self.audio.channels == 4:
                stereo = stereo_monitor_downmix(decode_foa(self.audio.samples, convention))
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
        self.statusBar().showMessage("Analyzing the sound field…" if kind == "analyze" else "Generating architecture…")
        self._worker.start()

    def _job_ok(self, result) -> None:
        self._end_busy()
        if self._closing:
            return
        if self._job == "analyze":
            self.field = result
            self._signature = self._pending_signature
            self.mesh = None
            self.report = None
            self.viewport.set_mesh(None)
            self._shown_frame = -1
            self._refresh_views()
            self.analysis_panel.show_report(None, self.field)
            self.analysis_panel.show_messages(result.warnings)
            self.statusBar().showMessage("Analysis complete. Adjust the design, then generate architecture.")
            return
        mesh, _response, used, warnings, report = result
        self.mesh = mesh
        self.report = report
        self.geometry_panel.set_parameters(used)
        self.viewport.set_mesh(mesh)
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
            self.viewport.set_overlay(None, None, params)
            return
        settings = self.audio_panel.settings()
        time_s = self.player.position_seconds if settings.view_time_mode == "frame" else None
        values = self.field.display_map(settings.view_band, time_s, settings.view_smoothing)
        self.viewport.set_overlay(self.field.azimuth_deg, values.max(axis=0), params)

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
        try:
            save_project(
                path,
                None if self.audio is None else self.audio.path,
                self.audio_panel.current_convention(),
                self.audio_panel.settings(),
                self.geometry_panel.parameters(),
                self.field,
                self.mesh,
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
        self.geometry_panel.set_parameters(record.design)
        self.audio_panel.apply_settings(record.analysis_settings)
        self.field = record.sound_field
        self.mesh = record.mesh
        self.report = None
        self.audio = None
        messages = list(record.warnings)
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
        self.viewport.set_mesh(self.mesh)
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
        answer = QMessageBox.question(self, "Reset project", "Clear the recording, analysis, mesh, and design parameters?")
        if answer != QMessageBox.Yes:
            return
        self.player.stop()
        self.audio = None
        self.field = None
        self.mesh = None
        self.report = None
        self._signature = None
        self.audio_panel.clear()
        self.geometry_panel.set_parameters(default_parameters())
        self.viewport.set_mesh(None)
        self.analysis_panel.clear()
        self._refresh_guides()
        self.statusBar().showMessage("Project reset.")

    def closeEvent(self, event) -> None:
        self._closing = True
        self.player.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(1500)
        self.viewport.shutdown()
        super().closeEvent(event)
