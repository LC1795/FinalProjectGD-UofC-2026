"""Synthetic first-order Ambisonic signals for tests and the in-app example.

Encoding follows the same conventions as the decoder:

- Azimuth 0 is microphone forward, positive toward microphone left.
- Elevation is positive upward.
- SN3D plane wave: W = s, X = s·cos(el)·cos(az), Y = s·cos(el)·sin(az), Z = s·sin(el).
- ACN storage order is W, Y, Z, X.
- FuMa storage order is W, X, Y, Z with W divided by √2.
- N3D multiplies X, Y, Z by √3 and keeps ACN order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from soundform.audio.format_detection import append_axml_chunk

_AMBIX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ambisonic>
  <format>ambix</format>
  <order>1</order>
  <ordering>ACN</ordering>
  <normalization>SN3D</normalization>
  <note>Synthetic Soundform example. Not a field recording.</note>
</ambisonic>
"""


def encode_plane_wave(
    azimuth_deg: float,
    elevation_deg: float,
    signal: np.ndarray,
    convention: str,
) -> np.ndarray:
    """Encode a monophonic pressure signal as a unit-direction FOA plane wave."""
    waveform = np.asarray(signal, dtype=np.float64).reshape(-1)
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    forward = np.cos(elevation) * np.cos(azimuth)
    left = np.cos(elevation) * np.sin(azimuth)
    up = np.sin(elevation)
    if convention == "acn_sn3d":
        channels = (waveform, left * waveform, up * waveform, forward * waveform)
    elif convention == "acn_n3d":
        scale = np.sqrt(3.0)
        channels = (
            waveform,
            scale * left * waveform,
            scale * up * waveform,
            scale * forward * waveform,
        )
    elif convention == "fuma":
        channels = (
            waveform / np.sqrt(2.0),
            forward * waveform,
            left * waveform,
            up * waveform,
        )
    else:
        raise ValueError(f"Unsupported convention: {convention}")
    return np.column_stack(channels)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> None:
    sf.write(str(path), np.asarray(samples), int(sample_rate), subtype=subtype)


def write_synthetic_example(path: str | Path, sample_rate: int = 48000) -> dict[str, str]:
    """Write a short AmbiX example: front, then left-elevated.

    The file is generated locally. It is not taken from a published dataset.
    """
    file_path = Path(path)
    duration_a = 1.5
    duration_b = 1.5
    count_a = int(sample_rate * duration_a)
    count_b = int(sample_rate * duration_b)
    rng = np.random.default_rng(7)
    time_a = np.arange(count_a) / sample_rate
    time_b = np.arange(count_b) / sample_rate
    front = 0.25 * rng.normal(0.0, 1.0, count_a) + 0.15 * np.sin(2 * np.pi * 180 * time_a)
    left = 0.22 * rng.normal(0.0, 1.0, count_b) + 0.12 * np.sin(2 * np.pi * 2400 * time_b)
    window_a = np.hanning(count_a)
    window_b = np.hanning(count_b)
    part_a = encode_plane_wave(0.0, 0.0, front * window_a, "acn_sn3d")
    part_b = encode_plane_wave(90.0, 20.0, left * window_b, "acn_sn3d")
    samples = np.vstack([part_a, part_b])
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples *= 0.5 / peak
    write_wav(file_path, samples.astype(np.float32), sample_rate, subtype="PCM_24")
    append_axml_chunk(file_path, _AMBIX_XML)
    return {
        "path": str(file_path),
        "convention": "acn_sn3d",
        "description": (
            "Synthetic ACN/SN3D example. From 0 to 1.5 s the source is at azimuth 0° "
            "(microphone forward) and elevation 0°. From 1.5 to 3.0 s it is at azimuth "
            "90° (microphone left) and elevation 20°. This is not a field recording."
        ),
    }
