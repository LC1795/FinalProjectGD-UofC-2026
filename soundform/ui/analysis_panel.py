"""Spectrum, directional map, and heuristic before/after comparison."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget

from soundform.acoustics.protection_estimator import ProtectionReport
from soundform.acoustics.sound_field import SoundField
from soundform.ui.theme import apply_plot_style


class AnalysisPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        apply_plot_style()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.summary = QLabel("Import a first-order Ambisonic WAV and press Analyze Audio.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.spectrum_canvas = self._canvas()
        self.direction_canvas = self._canvas()
        self.protection_canvas = self._canvas()
        self.tabs.addTab(self.spectrum_canvas, "Spectrum and timeline")
        self.tabs.addTab(self.direction_canvas, "Directional energy")
        self.tabs.addTab(self.protection_canvas, "Protection comparison")
        layout.addWidget(self.tabs, stretch=1)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setPlaceholderText("Analysis notes and heuristic warnings appear here.")
        self.notes.setFixedHeight(108)
        layout.addWidget(self.notes)
        self._blank(self.spectrum_canvas, "Spectrum appears after analysis.")
        self._blank(self.direction_canvas, "The directional map appears after analysis.")
        self._blank(self.protection_canvas, "Generate architecture to compare heuristic exposure.")

    def _canvas(self) -> FigureCanvas:
        figure = Figure(figsize=(6.2, 2.6), constrained_layout=True)
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(180)
        return canvas

    def show_messages(self, messages: list[str]) -> None:
        self.notes.setPlainText("\n\n".join(messages))

    def show_field(self, field: SoundField, band: str, smoothing: float, time_s: float | None) -> None:
        frame_weights = np.maximum(field.bin_confidence, 1e-3)
        confidence = float(np.average(field.bin_confidence, weights=frame_weights))
        agreement = float(np.average(field.agreement, weights=frame_weights))
        if field.silent:
            self.summary.setText(
                f"W level {field.rms_dbfs:.1f} dBFS in the analysed interval. "
                "The interval is effectively silent, so direction is undefined. Not dB SPL."
            )
        else:
            self.summary.setText(
                f"W level {field.rms_dbfs:.1f} dBFS (relative, not dB SPL). "
                f"Directional confidence {confidence:.2f}. Agreement {agreement:.2f}. "
                "Confidence is how plane-wave-like each tile is. Agreement is low when several directions are active together."
            )
        self._draw_spectrum(field, band)
        self._draw_direction(field, band, smoothing, time_s)
        self.spectrum_canvas.draw_idle()
        self.direction_canvas.draw_idle()

    def show_report(self, report: ProtectionReport | None, field: SoundField | None) -> None:
        figure = self.protection_canvas.figure
        figure.clear()
        if report is None or field is None:
            ax = figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Generate architecture to compare heuristic exposure.", ha="center", va="center")
            self.protection_canvas.draw_idle()
            return
        horizontal = np.clip(np.cos(np.deg2rad(report.elevation_deg)), 0.0, None)
        before = (report.exposure_before * horizontal[:, None]).sum(axis=0)
        after = (report.exposure_after * horizontal[:, None]).sum(axis=0)
        scale = max(float(np.max(before)), 1e-12)
        before = before / scale
        after = after / scale
        azimuth = report.azimuth_deg
        left = figure.add_subplot(121)
        left.plot(azimuth, before, color="#c4a574", label="Before")
        left.plot(azimuth, after, color="#7f92b5", label="After")
        left.set_xlim(0, 360)
        left.set_xlabel("Azimuth (0° mic forward)")
        left.set_ylabel("Relative exposure")
        left.set_title("Heuristic exposure")
        left.legend(frameon=False)
        right = figure.add_subplot(122)
        names = list(report.band_exposure)
        if names:
            index = np.arange(len(names))
            before_bars = [report.band_exposure[name]["before"] for name in names]
            after_bars = [report.band_exposure[name]["after"] for name in names]
            right.bar(index - 0.15, before_bars, width=0.3, color="#c4a574", label="Before")
            right.bar(index + 0.15, after_bars, width=0.3, color="#7f92b5", label="After")
            right.set_xticks(index, names)
            right.set_title("Band mean, same coverage")
            right.legend(frameon=False)
        else:
            right.set_axis_off()
        figure.suptitle("Occlusion heuristic, not a decibel reduction")
        self.protection_canvas.draw_idle()

    def clear(self) -> None:
        self.summary.setText("Import a first-order Ambisonic WAV and press Analyze Audio.")
        self.notes.clear()
        self._blank(self.spectrum_canvas, "Spectrum appears after analysis.")
        self._blank(self.direction_canvas, "The directional map appears after analysis.")
        self._blank(self.protection_canvas, "Generate architecture to compare heuristic exposure.")

    def _draw_spectrum(self, field: SoundField, band: str) -> None:
        figure = self.spectrum_canvas.figure
        figure.clear()
        spectrum = figure.add_subplot(211)
        spectrum.plot(field.freq_hz, field.spectrum_db_rel, color="#e0d2b8", linewidth=1.0)
        colors = {"low": "#7f92b5", "mid": "#c4a574", "high": "#c47b62"}
        for name, limits in field.band_limits.items():
            if limits[1] <= limits[0]:
                continue
            spectrum.axvspan(limits[0], limits[1], color=colors[name], alpha=0.18, label=name)
        spectrum.set_xlim(max(20.0, float(field.freq_hz[0]) if field.freq_hz.size else 20.0), max(field.freq_hz.max() if field.freq_hz.size else 1000, 100))
        spectrum.set_ylabel("dB re peak")
        spectrum.set_title("Relative spectrum of decoded W. Not dB SPL.")
        spectrum.legend(frameon=False, ncol=3)
        timeline = figure.add_subplot(212)
        finite = np.isfinite(field.dominant_az)
        if np.any(finite):
            timeline.scatter(
                field.times[finite],
                field.dominant_az[finite],
                c=field.bin_confidence[finite],
                cmap="magma",
                s=12,
                vmin=0,
                vmax=1,
            )
        timeline.set_ylim(0, 360)
        timeline.set_xlabel("Time (s)")
        timeline.set_ylabel("Azimuth (°)")
        timeline.set_title("Net direction per window. Gaps mean the bins disagree or the window is quiet.")
        if band != "broadband":
            timeline.text(0.01, 1.02, f"Timeline is broadband. The map tab follows {band}.", transform=timeline.transAxes, fontsize=8)

    def _draw_direction(self, field: SoundField, band: str, smoothing: float, time_s: float | None) -> None:
        figure = self.direction_canvas.figure
        figure.clear()
        axis = figure.add_subplot(111)
        values = field.display_map(band, time_s, smoothing)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            extent=(0.0, 360.0, -90.0, 90.0),
        )
        figure.colorbar(image, ax=axis, label="Relative energy", fraction=0.046, pad=0.04)
        axis.set_xlabel("Azimuth: 0° microphone forward, positive toward microphone left")
        axis.set_ylabel("Elevation (degrees)")
        moment = "analysed interval" if time_s is None else f"{time_s:.2f} s"
        axis.set_title(f"{band} directional energy, {moment}. Equirectangular, not a site plan.")

    def _blank(self, canvas: FigureCanvas, message: str) -> None:
        figure = canvas.figure
        figure.clear()
        axis = figure.add_subplot(111)
        axis.set_axis_off()
        axis.text(0.5, 0.5, message, ha="center", va="center")
        canvas.draw_idle()
