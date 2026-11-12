"""Turn a directional sound field into a per-direction design score.

Every component is scaled by its own peak so a weight of 1 means "use the full
range of that component", not a raw energy unit. Components are then combined
with the user's weights and divided by the weight sum. Missing energy, a zero
weight sum, or a silent field produces a zero score rather than a NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soundform.acoustics.sound_field import SoundField


@dataclass
class ResponseField:
    """Spherical score used by the form generators and the comparison panel."""

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    energy: np.ndarray
    score: np.ndarray
    azimuth_score: np.ndarray
    persistence: np.ndarray
    transient: np.ndarray
    confidence: np.ndarray
    weights: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _peak_normalize(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(array)) if array.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(array)
    return np.clip(array / peak, 0.0, 1.0)


def _time_stats(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Persistence and transient intensity on a (time, elevation, azimuth) stack."""
    if series.shape[0] == 0:
        shape = series.shape[1:]
        return np.zeros(shape), np.zeros(shape)
    peak = np.max(series, axis=0)
    mean = np.mean(series, axis=0)
    persistence = mean / (peak + 1e-20)
    if series.shape[0] == 1:
        transient = np.zeros(peak.shape, dtype=np.float64)
    else:
        high = np.percentile(series, 95, axis=0)
        median = np.median(series, axis=0)
        transient = np.clip((high - median) / (high + 1e-20), 0.0, 1.0)
    return persistence, transient


def _horizontal_score(score: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Collapse elevation with a cos(el) weight so overhead energy does not build a ridge."""
    weights = np.clip(np.cos(np.deg2rad(elevation_deg)), 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.zeros(score.shape[1], dtype=np.float64)
    collapsed = (score * weights[:, None]).sum(axis=0) / total
    return _peak_normalize(collapsed)


def compute_response(sound_field: SoundField, weights: dict[str, float], smoothing: float = 0.0) -> ResponseField:
    """Weighted directional score. `smoothing` is 0–1 and only affects the score."""
    from soundform.acoustics.directional_energy import smooth_azimuth_elevation

    warnings: list[str] = []
    named = {
        "broadband": max(0.0, float(weights.get("broadband", 0.0))),
        "low": max(0.0, float(weights.get("low", 0.0))),
        "mid": max(0.0, float(weights.get("mid", 0.0))),
        "high": max(0.0, float(weights.get("high", 0.0))),
        "persistence": max(0.0, float(weights.get("persistence", 0.0))),
        "transient": max(0.0, float(weights.get("transient", 0.0))),
        "confidence": max(0.0, float(weights.get("confidence", 0.0))),
    }
    weight_sum = sum(named.values())
    broadband = np.asarray(sound_field.broadband, dtype=np.float64)
    low = np.asarray(sound_field.low, dtype=np.float64)
    mid = np.asarray(sound_field.mid, dtype=np.float64)
    high = np.asarray(sound_field.high, dtype=np.float64)

    persistence, transient = _time_stats(broadband)
    energy_stack = (
        named["broadband"] * _peak_normalize(broadband.mean(axis=0))
        + named["low"] * _peak_normalize(low.mean(axis=0))
        + named["mid"] * _peak_normalize(mid.mean(axis=0))
        + named["high"] * _peak_normalize(high.mean(axis=0))
    )
    band_weight = named["broadband"] + named["low"] + named["mid"] + named["high"]
    if band_weight <= 1e-12:
        energy = np.zeros(broadband.shape[1:], dtype=np.float64)
    else:
        energy = energy_stack / band_weight

    confidence = np.asarray(sound_field.cell_confidence, dtype=np.float64)
    if confidence.shape[0] == broadband.shape[0]:
        weight_frames = np.maximum(broadband.mean(axis=(1, 2)), 1e-20)
        confidence_map = np.average(confidence, axis=0, weights=weight_frames)
    else:
        confidence_map = np.zeros(broadband.shape[1:], dtype=np.float64)

    if sound_field.silent or weight_sum <= 1e-12:
        if weight_sum <= 1e-12:
            warnings.append("All response weights are zero, so the form score is flat.")
        zeros = np.zeros(broadband.shape[1:], dtype=np.float64)
        return ResponseField(
            azimuth_deg=np.asarray(sound_field.azimuth_deg, dtype=np.float64),
            elevation_deg=np.asarray(sound_field.elevation_deg, dtype=np.float64),
            energy=zeros,
            score=zeros,
            azimuth_score=np.zeros(broadband.shape[2], dtype=np.float64),
            persistence=_peak_normalize(persistence),
            transient=_peak_normalize(transient),
            confidence=_peak_normalize(confidence_map),
            weights=named,
            warnings=warnings,
        )

    combined = (
        named["broadband"] * _peak_normalize(broadband.mean(axis=0))
        + named["low"] * _peak_normalize(low.mean(axis=0))
        + named["mid"] * _peak_normalize(mid.mean(axis=0))
        + named["high"] * _peak_normalize(high.mean(axis=0))
        + named["persistence"] * _peak_normalize(persistence)
        + named["transient"] * _peak_normalize(transient)
        + named["confidence"] * _peak_normalize(confidence_map)
    ) / weight_sum
    score = _peak_normalize(combined)
    sigma = max(0.0, float(smoothing)) * 4.0
    score = smooth_azimuth_elevation(score, sigma)
    score = _peak_normalize(score)
    return ResponseField(
        azimuth_deg=np.asarray(sound_field.azimuth_deg, dtype=np.float64),
        elevation_deg=np.asarray(sound_field.elevation_deg, dtype=np.float64),
        energy=_peak_normalize(energy),
        score=score,
        azimuth_score=_horizontal_score(score, sound_field.elevation_deg),
        persistence=_peak_normalize(persistence),
        transient=_peak_normalize(transient),
        confidence=_peak_normalize(confidence_map),
        weights=named,
        warnings=warnings,
    )
