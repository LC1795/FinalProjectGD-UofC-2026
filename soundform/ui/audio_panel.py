"""Import, inspection, playback, and analysis controls."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QHeaderView,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from soundform.audio.format_detection import CONVENTION_HELP, CONVENTION_LABELS
from soundform.audio.loader import AudioFile
from soundform.models.project import AnalysisSettings
from soundform.ui.theme import apply_plot_style

_CHANNEL_NAMES = {
    "acn_sn3d": ("W", "Y", "Z", "X"),
    "fuma": ("W", "X", "Y", "Z"),
    "acn_n3d": ("W", "Y", "Z", "X"),
}


class AudioPanel(QWidget):
    analyze_requested = Signal()
    convention_changed = Signal(str)
    playback_toggled = Signal()
    playback_stopped = Signal()
    seek_requested = Signal(float)
    scrub_started = Signal()
    display_changed = Signal()
    analysis_inputs_changed = Signal()
    open_requested = Signal(str)
    example_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_plot_style()
        self.audio: AudioFile | None = None
        self._duration = 0.0
        self._scrubbing = False
        self._playhead = None
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Audio")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        root.addWidget(title)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("Import WAV")
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(lambda *_args: self._import())
        self.example_button = QPushButton("Example")
        self.example_button.setToolTip("Write a synthetic first-order Ambisonic WAV with a known direction change.")
        self.example_button.clicked.connect(lambda *_args: self._example())
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.example_button)
        root.addLayout(buttons)

        self.meta = QLabel("No recording loaded.")
        self.meta.setWordWrap(True)
        root.addWidget(self.meta)
        self.validation = QLabel("Import a WAV file to begin.")
        self.validation.setObjectName("validation")
        self.validation.setWordWrap(True)
        root.addWidget(self.validation)
        self.permanent = QLabel(
            "Levels are relative to full scale, not calibrated dB SPL. "
            "The model is an abstract design, not the room the recording was made in."
        )
        self.permanent.setObjectName("hint")
        self.permanent.setWordWrap(True)
        root.addWidget(self.permanent)

        convention_row = QHBoxLayout()
        self.convention = QComboBox()
        self.convention.addItem("Choose a convention…", None)
        for key, label in CONVENTION_LABELS.items():
            self.convention.addItem(label, key)
        self.convention.currentIndexChanged.connect(lambda *_args: self._on_convention())
        help_button = QPushButton("Help")
        help_button.setToolTip("How ACN/SN3D, FuMa, and ACN/N3D order the four channels.")
        help_button.clicked.connect(lambda *_args: self._help())
        convention_row.addWidget(self.convention, stretch=1)
        convention_row.addWidget(help_button)
        root.addLayout(convention_row)

        self.channels = QTableWidget(0, 4)
        self.channels.setHorizontalHeaderLabels(["#", "Name", "RMS", "Peak"])
        header = self.channels.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setToolTip("RMS and peak are dBFS, relative to digital full scale. They are not dB SPL.")
        self.channels.setSelectionBehavior(QTableWidget.SelectRows)
        self.channels.setSelectionMode(QTableWidget.SingleSelection)
        self.channels.verticalHeader().setVisible(False)
        self.channels.setMinimumHeight(120)
        self.channels.itemSelectionChanged.connect(self._redraw_waveform)
        root.addWidget(self.channels)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(lambda *_args: self.playback_toggled.emit())
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(lambda *_args: self.playback_stopped.emit())
        self.time_label = QLabel("0.00 / 0.00 s")
        transport.addWidget(self.play_button)
        transport.addWidget(self.stop_button)
        transport.addWidget(self.time_label)
        root.addLayout(transport)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderPressed.connect(self._begin_scrub)
        self.slider.sliderReleased.connect(self._end_scrub)
        self.slider.valueChanged.connect(self._preview_scrub)
        root.addWidget(self.slider)
        self.monitor_note = QLabel("Playback is a monitor, not the spatial analysis.")
        self.monitor_note.setObjectName("hint")
        self.monitor_note.setWordWrap(True)
        root.addWidget(self.monitor_note)

        interval = QGroupBox("Analysis interval")
        interval_form = QFormLayout(interval)
        self.start_spin = self._seconds_spin()
        self.end_spin = self._seconds_spin()
        self.start_spin.valueChanged.connect(lambda *_args: self.analysis_inputs_changed.emit())
        self.end_spin.valueChanged.connect(lambda *_args: self.analysis_inputs_changed.emit())
        self.start_spin.valueChanged.connect(lambda _value: self._redraw_waveform())
        self.end_spin.valueChanged.connect(lambda _value: self._redraw_waveform())
        interval_form.addRow("Start (s)", self.start_spin)
        interval_form.addRow("End (s)", self.end_spin)
        root.addWidget(interval)

        bands = QGroupBox("Frequency bands (Hz)")
        band_form = QFormLayout(bands)
        self.low_min, self.low_max = self._band_row(band_form, "Low", 20, 250)
        self.mid_min, self.mid_max = self._band_row(band_form, "Mid", 250, 2000)
        self.high_min, self.high_max = self._band_row(band_form, "High", 2000, 12000)
        root.addWidget(bands)

        view = QGroupBox("Visualization")
        view_form = QFormLayout(view)
        self.view_band = QComboBox()
        self.view_band.addItem("Broadband", "broadband")
        self.view_band.addItem("Low", "low")
        self.view_band.addItem("Mid", "mid")
        self.view_band.addItem("High", "high")
        self.view_band.currentIndexChanged.connect(lambda *_args: self.display_changed.emit())
        self.time_mode = QComboBox()
        self.time_mode.addItem("Whole analysis interval", "aggregate")
        self.time_mode.addItem("Frame at the playhead", "frame")
        self.time_mode.currentIndexChanged.connect(lambda *_args: self.display_changed.emit())
        self.smooth = QSlider(Qt.Horizontal)
        self.smooth.setRange(0, 100)
        self.smooth.setValue(20)
        self.smooth.valueChanged.connect(lambda *_args: self.display_changed.emit())
        view_form.addRow("Band", self.view_band)
        view_form.addRow("Time", self.time_mode)
        view_form.addRow("Smoothing", self.smooth)
        root.addWidget(view)

        self.analyze_button = QPushButton("Analyze Audio")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(lambda *_args: self.analyze_requested.emit())
        root.addWidget(self.analyze_button)

        self.figure = Figure(figsize=(3.4, 1.8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(150)
        root.addWidget(self.canvas)
        self._blank_waveform()
        root.addStretch(1)

    def show_audio(self, audio: AudioFile) -> None:
        self.audio = audio
        self._duration = audio.duration
        self.meta.setText(
            f"{audio.name}\n"
            f"{audio.duration:.2f} s · {audio.sample_rate} Hz · {audio.channels} ch · {audio.bit_depth}"
        )
        self._set_validation(audio.detection.messages, audio.detection.severity)
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setRange(0.0, max(0.0, audio.duration))
        self.end_spin.setRange(0.0, max(0.0, audio.duration))
        self.start_spin.setValue(0.0)
        self.end_spin.setValue(audio.duration)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)
        suggestion = audio.detection.suggested_convention
        self.convention.blockSignals(True)
        if suggestion:
            index = self.convention.findData(suggestion)
            self.convention.setCurrentIndex(max(index, 0))
        elif audio.detection.spatial_status != "ready":
            self.convention.setCurrentIndex(0)
        self.convention.setEnabled(audio.detection.can_force_convention)
        self.convention.blockSignals(False)
        self._fill_channels()
        self._sync_analyze()
        self.slider.setValue(0)
        self.time_label.setText(f"0.00 / {audio.duration:.2f} s")
        self._redraw_waveform()

    def apply_settings(self, settings: AnalysisSettings) -> None:
        blockers = (
            self.start_spin,
            self.end_spin,
            self.low_min,
            self.low_max,
            self.mid_min,
            self.mid_max,
            self.high_min,
            self.high_max,
            self.view_band,
            self.time_mode,
            self.smooth,
        )
        for widget in blockers:
            widget.blockSignals(True)
        self.start_spin.setValue(settings.start_s)
        if settings.end_s > 0:
            self.end_spin.setValue(settings.end_s)
        self.low_min.setValue(settings.low_min_hz)
        self.low_max.setValue(settings.low_max_hz)
        self.mid_min.setValue(settings.mid_min_hz)
        self.mid_max.setValue(settings.mid_max_hz)
        self.high_min.setValue(settings.high_min_hz)
        self.high_max.setValue(settings.high_max_hz)
        band_index = self.view_band.findData(settings.view_band)
        if band_index >= 0:
            self.view_band.setCurrentIndex(band_index)
        mode_index = self.time_mode.findData(settings.view_time_mode)
        if mode_index >= 0:
            self.time_mode.setCurrentIndex(mode_index)
        self.smooth.setValue(int(round(settings.view_smoothing * 100)))
        for widget in blockers:
            widget.blockSignals(False)

    def set_convention(self, convention: str | None) -> None:
        self.convention.blockSignals(True)
        if convention is None:
            self.convention.setCurrentIndex(0)
        else:
            index = self.convention.findData(convention)
            if index >= 0:
                self.convention.setCurrentIndex(index)
        self.convention.blockSignals(False)
        self._fill_channels()
        self._sync_analyze()
        self._redraw_waveform()

    def current_convention(self) -> str | None:
        return self.convention.currentData()

    def settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            start_s=float(self.start_spin.value()),
            end_s=float(self.end_spin.value()),
            low_min_hz=float(self.low_min.value()),
            low_max_hz=float(self.low_max.value()),
            mid_min_hz=float(self.mid_min.value()),
            mid_max_hz=float(self.mid_max.value()),
            high_min_hz=float(self.high_min.value()),
            high_max_hz=float(self.high_max.value()),
            view_band=self.view_band.currentData(),
            view_smoothing=self.smooth.value() / 100.0,
            view_time_mode=self.time_mode.currentData(),
        )

    def set_playing(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")

    def set_monitor_note(self, text: str) -> None:
        self.monitor_note.setText(text)

    def set_position(self, seconds: float) -> None:
        if self._scrubbing or self._duration <= 0:
            return
        self.slider.blockSignals(True)
        self.slider.setValue(int(np.clip(seconds / self._duration, 0.0, 1.0) * 10000))
        self.slider.blockSignals(False)
        self.time_label.setText(f"{seconds:.2f} / {self._duration:.2f} s")
        self._move_playhead(seconds)

    def set_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy and self._can_analyze())
        self.import_button.setEnabled(not busy)

    def position_seconds(self) -> float:
        if self._duration <= 0:
            return 0.0
        return (self.slider.value() / 10000.0) * self._duration

    def _import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import WAV", "", "WAV audio (*.wav *.wave)")
        if path:
            self.open_requested.emit(path)

    def _example(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Save synthetic FOA example", "soundform-example.wav", "WAV audio (*.wav)")
        if path:
            self.example_requested.emit(path)

    def _help(self) -> None:
        QMessageBox.information(self, "Ambisonic conventions", CONVENTION_HELP)

    def _on_convention(self) -> None:
        self._fill_channels()
        self._sync_analyze()
        self._redraw_waveform()
        chosen = self.current_convention()
        if chosen:
            self.convention_changed.emit(chosen)
        self.analysis_inputs_changed.emit()

    def _sync_analyze(self) -> None:
        self.analyze_button.setEnabled(self._can_analyze())

    def _can_analyze(self) -> bool:
        if self.audio is None or not self.audio.detection.can_force_convention:
            return False
        return self.current_convention() is not None

    def _fill_channels(self) -> None:
        if self.audio is None:
            self.channels.setRowCount(0)
            return
        names = _CHANNEL_NAMES.get(self.current_convention() or "", tuple(f"Channel {i + 1}" for i in range(self.audio.channels)))
        if len(names) != self.audio.channels:
            names = tuple(f"Channel {i + 1}" for i in range(self.audio.channels))
        self.channels.setRowCount(self.audio.channels)
        for row, stats in enumerate(self.audio.channel_stats):
            values = (str(row + 1), names[row], f"{stats.rms_dbfs:.1f}", f"{stats.peak_dbfs:.1f}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.channels.setItem(row, column, item)
        self.channels.selectRow(0)

    def _set_validation(self, messages: list[str], severity: str) -> None:
        colors = {"error": "#6a3030", "warning": "#6a5428", "info": "#2c3a30"}
        self.validation.setText(" ".join(messages))
        self.validation.setStyleSheet(
            f"background: {colors.get(severity, '#2a2d2a')}; padding: 6px; border-radius: 4px;"
        )

    def _seconds_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        return spin

    def _band_row(self, form: QFormLayout, label: str, default_min: float, default_max: float):
        low = QDoubleSpinBox()
        high = QDoubleSpinBox()
        for spin, value in ((low, default_min), (high, default_max)):
            spin.setRange(0.0, 96000.0)
            spin.setDecimals(0)
            spin.setSingleStep(10)
            spin.setValue(value)
            spin.valueChanged.connect(lambda *_args: self.analysis_inputs_changed.emit())
        row = QHBoxLayout()
        row.addWidget(low)
        row.addWidget(QLabel("to"))
        row.addWidget(high)
        host = QWidget()
        host.setLayout(row)
        form.addRow(label, host)
        return low, high

    def _begin_scrub(self) -> None:
        self._scrubbing = True
        self.scrub_started.emit()

    def _preview_scrub(self, _value: int) -> None:
        seconds = self.position_seconds()
        self.time_label.setText(f"{seconds:.2f} / {self._duration:.2f} s")
        self._move_playhead(seconds)

    def _end_scrub(self) -> None:
        self._scrubbing = False
        self.seek_requested.emit(self.position_seconds())

    def _redraw_waveform(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        if self.audio is None:
            axis.set_axis_off()
            axis.text(0.5, 0.5, "Waveform", ha="center", va="center")
            self.canvas.draw_idle()
            return
        row = self.channels.currentRow()
        if row < 0:
            row = 0
        channel = self.audio.samples[:, row]
        buckets = 1400
        if channel.size > buckets:
            usable = channel.size - (channel.size % buckets)
            shaped = channel[:usable].reshape(buckets, -1)
            low = shaped.min(axis=1)
            high = shaped.max(axis=1)
            times = np.linspace(0.0, usable / self.audio.sample_rate, buckets)
        else:
            times = np.arange(channel.size) / self.audio.sample_rate
            low = high = channel
        axis.fill_between(times, low, high, color="#c4a574", linewidth=0)
        axis.axvspan(self.start_spin.value(), self.end_spin.value(), color="#7f92b5", alpha=0.15)
        self._playhead = axis.axvline(self.position_seconds(), color="#e7e2d6", linewidth=1.0)
        axis.set_xlim(0.0, max(self._duration, 0.01))
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
        axis.set_title("Selected channel, file order. dBFS in the table is not dB SPL.")
        self.canvas.draw_idle()

    def _move_playhead(self, seconds: float) -> None:
        if self._playhead is None:
            return
        self._playhead.set_xdata([seconds, seconds])
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.audio = None
        self._duration = 0.0
        self._playhead = None
        self.meta.setText("No recording loaded.")
        self._set_validation(["Import a WAV file to begin."], "info")
        self.convention.blockSignals(True)
        self.convention.setCurrentIndex(0)
        self.convention.setEnabled(True)
        self.convention.blockSignals(False)
        self.channels.setRowCount(0)
        self.analyze_button.setEnabled(False)
        self.slider.setValue(0)
        self.time_label.setText("0.00 / 0.00 s")
        self.set_playing(False)
        self.set_monitor_note("Playback is a monitor, not the spatial analysis.")
        self._redraw_waveform()

    def _blank_waveform(self) -> None:
        self._redraw_waveform()
