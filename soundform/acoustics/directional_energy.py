"""Directional energy from decoded first-order Ambisonics.

For SN3D W and Cartesian (X, Y, Z), a plane wave at the origin is real-valued
and in phase across components. The active-intensity vector

    I(f, t) = Re{ W(f, t) · conj([X, Y, Z](f, t)) }

then points toward the source, and |I| equals |W|² for a unit-direction plane
wave. Diffuseness of a bin is 1 − |I| / |W|².

Each time-frequency bin contributes its direct energy |W|² · confidence to the
spherical cell of that arrival, and its diffuse remainder uniformly in
solid angle. First-order tiles still carry only one arrival: overlapping
sources in the same bin are not separated. The result is a directional energy
distribution, not a single angle and not a geometric map.
"""

from __future__ import annotations

import numpy as np

# Equal-angle grids. Cell area uses the exact sin(el) solid-angle factor so a
# diffuse field is flat when drawn on this grid.
N_AZIMUTH = 72
N_ELEVATION = 18


def grid_axes(n_azimuth: int = N_AZIMUTH, n_elevation: int = N_ELEVATION) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuth bin centres, elevation bin centres, and elevation edges in degrees."""
    azimuth = (np.arange(n_azimuth) + 0.5) * (360.0 / n_azimuth)
    edges = np.linspace(-90.0, 90.0, n_elevation + 1)
    elevation = 0.5 * (edges[:-1] + edges[1:])
    return azimuth, elevation, edges


def cell_solid_angle_share(elevation_edges: np.ndarray, n_azimuth: int) -> np.ndarray:
    """Fraction of the sphere in each cell, shaped (elevation, azimuth)."""
    sin_hi = np.sin(np.deg2rad(elevation_edges[1:]))
    sin_lo = np.sin(np.deg2rad(elevation_edges[:-1]))
    delta_az = np.deg2rad(360.0 / n_azimuth)
    cell = (sin_hi - sin_lo) * delta_az
    share = cell / (4.0 * np.pi)
    return np.broadcast_to(share[:, None], (share.shape[0], n_azimuth)).copy()


def active_intensity(
    pressure: np.ndarray,
    forward: np.ndarray,
    left: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Ix, Iy, Iz, energy |W|², and per-bin confidence |I|/|W|²."""
    energy = np.abs(pressure) ** 2
    intensity_x = np.real(pressure * np.conj(forward))
    intensity_y = np.real(pressure * np.conj(left))
    intensity_z = np.real(pressure * np.conj(up))
    magnitude = np.sqrt(intensity_x**2 + intensity_y**2 + intensity_z**2)
    confidence = np.zeros(energy.shape, dtype=np.float64)
    usable = energy > 1e-20
    np.divide(magnitude, energy, out=confidence, where=usable)
    np.clip(confidence, 0.0, 1.0, out=confidence)
    return intensity_x, intensity_y, intensity_z, energy, confidence


def accumulate_direction_histogram(
    intensity_x: np.ndarray,
    intensity_y: np.ndarray,
    intensity_z: np.ndarray,
    energy: np.ndarray,
    confidence: np.ndarray,
    frequency_mask: np.ndarray,
    n_azimuth: int,
    n_elevation: int,
    elevation_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram one frequency band into (time, elevation, azimuth).

    Returns the energy map and the direct-energy-weighted confidence accumulator.
    """
    if not np.any(frequency_mask):
        frames = energy.shape[1]
        shape = (frames, n_elevation, n_azimuth)
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)

    band_x = intensity_x[frequency_mask]
    band_y = intensity_y[frequency_mask]
    band_z = intensity_z[frequency_mask]
    band_energy = energy[frequency_mask]
    band_confidence = confidence[frequency_mask]
    direct = band_energy * band_confidence
    diffuse = band_energy * (1.0 - band_confidence)

    magnitude = np.sqrt(band_x**2 + band_y**2 + band_z**2)
    azimuth = np.degrees(np.arctan2(band_y, band_x))
    azimuth = np.mod(azimuth, 360.0)
    safe = np.maximum(magnitude, 1e-20)
    elevation = np.degrees(np.arcsin(np.clip(band_z / safe, -1.0, 1.0)))

    azimuth_index = np.floor(azimuth / (360.0 / n_azimuth)).astype(np.int32) % n_azimuth
    elev_span = float(elevation_edges[-1] - elevation_edges[0])
    elevation_index = np.floor((elevation - float(elevation_edges[0])) / (elev_span / n_elevation))
    elevation_index = np.clip(elevation_index, 0, n_elevation - 1).astype(np.int32)

    n_freq, n_frames = band_energy.shape
    frame_index = np.broadcast_to(np.arange(n_frames, dtype=np.int32), (n_freq, n_frames))
    histogram = np.zeros((n_frames, n_elevation, n_azimuth), dtype=np.float64)
    confidence_acc = np.zeros_like(histogram)
    np.add.at(
        histogram,
        (frame_index.ravel(), elevation_index.ravel(), azimuth_index.ravel()),
        direct.ravel(),
    )
    np.add.at(
        confidence_acc,
        (frame_index.ravel(), elevation_index.ravel(), azimuth_index.ravel()),
        (direct * band_confidence).ravel(),
    )
    share = cell_solid_angle_share(elevation_edges, n_azimuth)
    histogram += diffuse.sum(axis=0)[:, None, None] * share[None, :, :]
    return histogram, confidence_acc


def dominant_direction(
    intensity_x: np.ndarray,
    intensity_y: np.ndarray,
    intensity_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy-flow direction per frame, plus agreement |ΣI| / Σ|I|.

    Agreement falls when frequency bins point different ways, so the reported
    angle is withheld (NaN) instead of showing a meaningless average.
    """
    sum_x = intensity_x.sum(axis=0)
    sum_y = intensity_y.sum(axis=0)
    sum_z = intensity_z.sum(axis=0)
    net = np.sqrt(sum_x**2 + sum_y**2 + sum_z**2)
    spread = np.sqrt(intensity_x**2 + intensity_y**2 + intensity_z**2).sum(axis=0)
    agreement = net / (spread + 1e-20)
    azimuth = np.mod(np.degrees(np.arctan2(sum_y, sum_x)), 360.0)
    elevation = np.degrees(np.arcsin(np.clip(sum_z / np.maximum(net, 1e-20), -1.0, 1.0)))
    azimuth = azimuth.astype(np.float64)
    elevation = elevation.astype(np.float64)
    weak = agreement < 0.12
    azimuth[weak] = np.nan
    elevation[weak] = np.nan
    return azimuth, elevation, agreement.astype(np.float64)


def smooth_azimuth_elevation(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth a single (elevation, azimuth) map. Azimuth wraps."""
    if sigma <= 0.05:
        return values
    radius = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="wrap")
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    # Elevation does not wrap; use a reflected pad.
    el_pad = np.pad(smoothed, ((radius, radius), (0, 0)), mode="edge")
    result = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, el_pad)
    return result
