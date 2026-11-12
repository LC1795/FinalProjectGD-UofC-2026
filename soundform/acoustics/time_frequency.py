"""Short-time Fourier transforms for the four decoded Ambisonic channels."""

from __future__ import annotations

import numpy as np
from scipy import signal

from soundform.errors import AnalysisError


def choose_stft_params(sample_rate: int, n_samples: int) -> tuple[int, int, bool]:
    """Pick a window and hop.

    The hop stays at half the window until the file would produce more than
    1500 frames, then it coarsens. The coarsened flag is reported to the user;
    samples are not dropped silently without that notice.
    """
    if sample_rate <= 24000:
        nperseg = 2048
    elif sample_rate <= 48000:
        nperseg = 4096
    else:
        nperseg = 8192
    if n_samples < 256:
        raise AnalysisError("Select an analysis interval of at least 0.25 seconds.")
    if n_samples < nperseg:
        nperseg = int(2 ** np.floor(np.log2(n_samples)))
        nperseg = max(256, nperseg)
    hop = max(1, nperseg // 2)
    n_frames = 1 + max(0, (n_samples - nperseg) // hop)
    coarsened = False
    if n_frames > 1500:
        hop = max(hop, int(np.ceil((n_samples - nperseg) / 1500.0)))
        coarsened = True
    return nperseg, hop, coarsened


def stft_wxyz(
    wxyz: np.ndarray,
    sample_rate: int,
    nperseg: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies (Hz), frame-centre times (s), and spectra shaped (4, F, T)."""
    audio = np.asarray(wxyz, dtype=np.float64)
    if hop < 1 or nperseg < 16:
        raise AnalysisError("The STFT window is too short for this interval.")
    noverlap = nperseg - hop
    spectra = []
    frequencies = None
    times = None
    for channel in range(4):
        frequencies, times, values = signal.stft(
            audio[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            padded=False,
            boundary=None,
            detrend=False,
        )
        spectra.append(values)
    stacked = np.stack(spectra, axis=0)
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        stacked,
    )


def welch_relative_db(pressure: np.ndarray, sample_rate: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of W, in dB relative to its own peak. Not dB SPL."""
    count = int(pressure.shape[0])
    window = min(nperseg, count)
    if window < 32:
        return np.array([0.0]), np.array([-120.0])
    frequencies, density = signal.welch(
        pressure.astype(np.float64),
        fs=sample_rate,
        window="hann",
        nperseg=window,
        detrend=False,
    )
    peak = float(np.max(density)) if density.size else 0.0
    if peak <= 1e-20:
        relative = np.full(density.shape, -120.0)
    else:
        relative = 10.0 * np.log10(np.maximum(density, 1e-20) / peak)
    return frequencies.astype(np.float64), relative.astype(np.float64)
