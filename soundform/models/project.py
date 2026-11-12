"""Design parameters and the in-memory project.

Audio is stored as a filesystem path. Project JSON never contains audio samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class DesignParameters:
    """Tunable architectural response. Distances and heights are metres."""

    mode: str = "terrain"
    site_shape: str = "circle"
    site_radius: float = 30.0
    site_width: float = 60.0
    site_depth: float = 60.0
    max_height: float = 8.0
    min_height: float = 0.0
    grid_resolution: int = 64
    smoothing: float = 0.35
    response_intensity: float = 1.0
    weight_broadband: float = 0.30
    weight_low: float = 0.15
    weight_mid: float = 0.20
    weight_high: float = 0.10
    weight_persistence: float = 0.15
    weight_transient: float = 0.05
    weight_confidence: float = 0.05
    protected_radius: float = 5.0
    protected_center_x: float = 0.0
    protected_center_y: float = 0.0
    rotation_offset_deg: float = 0.0
    wall_thickness: float = 0.45
    barrier_spacing: float = 3.0
    curvature: float = 0.65
    openness: float = 0.35
    entry_width_deg: float = 28.0
    pavilion_diameter: float = 18.0
    roof_curvature: float = 0.55
    enclosure: float = 0.55
    opening_width_deg: float = 32.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignParameters":
        known = {item.name for item in fields(cls)}
        values = {key: data[key] for key in data if key in known}
        return cls(**values)

    def copy(self) -> "DesignParameters":
        return DesignParameters.from_dict(self.to_dict())


@dataclass
class AnalysisSettings:
    """Controls that change the analysed interval or the displayed band."""

    start_s: float = 0.0
    end_s: float = 0.0
    low_min_hz: float = 20.0
    low_max_hz: float = 250.0
    mid_min_hz: float = 250.0
    mid_max_hz: float = 2000.0
    high_min_hz: float = 2000.0
    high_max_hz: float = 12000.0
    view_band: str = "broadband"
    view_smoothing: float = 0.2
    view_time_mode: str = "aggregate"

    def band_limits(self) -> dict[str, tuple[float, float]]:
        return {
            "low": (float(self.low_min_hz), float(self.low_max_hz)),
            "mid": (float(self.mid_min_hz), float(self.mid_max_hz)),
            "high": (float(self.high_min_hz), float(self.high_max_hz)),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisSettings":
        known = {item.name for item in fields(cls)}
        values = {key: data[key] for key in data if key in known}
        return cls(**values)


def default_parameters() -> DesignParameters:
    return DesignParameters()


def default_analysis_settings() -> AnalysisSettings:
    return AnalysisSettings()
