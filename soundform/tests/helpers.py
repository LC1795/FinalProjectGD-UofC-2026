"""Shared synthetic fields and angle checks."""

from __future__ import annotations

import numpy as np

from soundform.acoustics.directional_energy import grid_axes
from soundform.acoustics.sound_field import LEVEL_NOTE, SoundField
from soundform.audio.synthetic import encode_plane_wave
from soundform.models.project import AnalysisSettings

BANDS = AnalysisSettings().band_limits()


def unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def angular_error_deg(azimuth_a, elevation_a, azimuth_b, elevation_b) -> float:
    first = unit_vector(azimuth_a, elevation_a)
    second = unit_vector(azimuth_b, elevation_b)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signal(kind: str, sample_rate: int, seconds: float, frequency: float, seed: int) -> np.ndarray:
    count = int(sample_rate * seconds)
    time = np.arange(count) / sample_rate
    if kind == "noise":
        waveform = np.random.default_rng(seed).normal(0.0, 0.2, count)
    else:
        waveform = np.sin(2.0 * np.pi * frequency * time)
    fade = min(count // 8, 256)
    if fade > 1:
        waveform = waveform.copy()
        waveform[:fade] *= np.linspace(0.0, 1.0, fade)
        waveform[-fade:] *= np.linspace(1.0, 0.0, fade)
    return waveform


def plane_wave(
    azimuth: float,
    elevation: float,
    *,
    convention: str = "acn_sn3d",
    seconds: float = 1.0,
    sample_rate: int = 16000,
    kind: str = "noise",
    frequency: float = 1000.0,
    seed: int = 1,
) -> tuple[np.ndarray, int]:
    waveform = signal(kind, sample_rate, seconds, frequency, seed)
    return encode_plane_wave(azimuth, elevation, waveform, convention), sample_rate


def blob_field(
    peak_azimuth: float = 0.0,
    peak_elevation: float = 0.0,
    *,
    n_time: int = 6,
    spread_deg: float = 16.0,
    transient_azimuth: float | None = None,
) -> SoundField:
    """A directional blob with known axes, without running an STFT."""
    azimuth, elevation, _edges = grid_axes()
    az_grid, el_grid = np.meshgrid(azimuth, elevation)
    peak = unit_vector(peak_azimuth, peak_elevation)
    cells = np.stack(
        [
            np.cos(np.deg2rad(el_grid)) * np.cos(np.deg2rad(az_grid)),
            np.cos(np.deg2rad(el_grid)) * np.sin(np.deg2rad(az_grid)),
            np.sin(np.deg2rad(el_grid)),
        ],
        axis=-1,
    )
    cosine = np.clip(cells @ peak, -1.0, 1.0)
    sigma = np.deg2rad(spread_deg)
    energy = np.exp((cosine - 1.0) / max(sigma**2, 1e-4))
    stack = np.repeat(energy.astype(np.float32)[None, ...], n_time, axis=0)
    if transient_azimuth is not None:
        other = unit_vector(transient_azimuth, 0.0)
        other_energy = np.exp((np.clip(cells @ other, -1.0, 1.0) - 1.0) / max(sigma**2, 1e-4))
        stack = stack.copy()
        stack[0] = np.maximum(stack[0], other_energy.astype(np.float32))
    times = np.linspace(0.15, 1.5, n_time)
    confidence = np.full(stack.shape, 0.92, dtype=np.float32)
    return SoundField(
        sample_rate=16000,
        convention="acn_sn3d",
        interval=(0.0, 1.6),
        duration=1.6,
        times=times,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        broadband=stack,
        low=stack.copy(),
        mid=stack.copy(),
        high=stack * 0.25,
        cell_confidence=confidence,
        bin_confidence=np.full(n_time, 0.9, dtype=np.float32),
        agreement=np.full(n_time, 0.85, dtype=np.float32),
        dominant_az=np.full(n_time, peak_azimuth, dtype=np.float64),
        dominant_el=np.full(n_time, peak_elevation, dtype=np.float64),
        frame_level_db=np.zeros(n_time, dtype=np.float32),
        freq_hz=np.array([100.0, 1000.0]),
        spectrum_db_rel=np.array([0.0, -6.0]),
        rms_dbfs=-18.0,
        peak_dbfs=-6.0,
        silent=False,
        warnings=[LEVEL_NOTE],
        band_limits=BANDS,
        stft_hop=512,
        stft_nperseg=1024,
    )
