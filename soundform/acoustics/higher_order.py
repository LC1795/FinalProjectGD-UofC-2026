"""Third-order ACN spherical harmonic steering and band-power estimation."""

from math import factorial

import numpy as np
from scipy.special import lpmv


def sn3d_basis(azimuth_deg, elevation_deg, order=3):
    """Real AmbiX basis in ACN order, with first four components W, Y, Z, X."""
    azimuth, elevation = np.broadcast_arrays(np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg))
    vertical = np.sin(elevation)
    channels = []
    for degree in range(order + 1):
        for harmonic in range(-degree, degree + 1):
            m = abs(harmonic)
            scale = np.sqrt((2 if m else 1) * factorial(degree - m) / factorial(degree + m))
            phase = np.sin(m * azimuth) if harmonic < 0 else np.cos(m * azimuth)
            channels.append((-1) ** m * scale * lpmv(m, degree, vertical) * phase)
    return np.stack(channels, axis=-1)


def beam_power_maps(spectra, masks, azimuth, elevation):
    """Scan band covariance with real spherical steering vectors.

    Input is N3D ACN. Normalize each map to the band's W energy so its
    values can be compared across frames and passed to the exposure estimator.
    """
    az_grid, el_grid = np.meshgrid(azimuth, elevation)
    n3d_scale = np.sqrt(2 * np.repeat(np.arange(4), [1, 3, 5, 7]) + 1)
    steering = (sn3d_basis(az_grid, el_grid) * n3d_scale).reshape(-1, 16).astype(np.float32)
    area = np.cos(np.deg2rad(el_grid)).ravel().astype(np.float32)
    # Project the symmetric real covariance using the triangular entries.
    row, col = np.triu_indices(16)
    projection = steering[:, row] * steering[:, col] * np.where(row == col, 1, 2).astype(np.float32)
    maps = {}
    for name, mask in masks.items():
        bins = spectra[:, mask, :].transpose(2, 0, 1)
        covariance = (bins.real @ bins.real.transpose(0, 2, 1)
                      + bins.imag @ bins.imag.transpose(0, 2, 1))
        power = np.maximum(covariance[:, row, col] @ projection.T, 0) * area
        total = power.sum(axis=1, keepdims=True)
        power *= np.divide(covariance[:, :1, 0], total, out=np.zeros_like(total), where=total > 0)
        maps[name] = power.reshape(-1, len(elevation), len(azimuth))
    return maps
