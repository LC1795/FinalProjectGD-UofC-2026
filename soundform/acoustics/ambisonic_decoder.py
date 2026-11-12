"""Decode supported first-order Ambisonic channel conventions to SN3D W, X, Y, Z.

Internal axes, matching the encoder:

- X: microphone forward
- Y: microphone left
- Z: up
- Azimuth is measured from forward toward left.
- Elevation is measured from the horizontal plane, positive upward.

FuMa W is multiplied by √2. ACN/N3D X, Y and Z are divided by √3. After this
conversion a unit plane wave satisfies W = s and (X, Y, Z) = s · direction.
"""

from __future__ import annotations

import numpy as np

from soundform.errors import AnalysisError

SUPPORTED_CONVENTIONS = ("acn_sn3d", "fuma", "acn_n3d")


def decode_foa(samples: np.ndarray, convention: str) -> np.ndarray:
    """Return float64 samples shaped (frames, 4) in SN3D order W, X, Y, Z."""
    if convention not in SUPPORTED_CONVENTIONS:
        raise AnalysisError(
            "Select a supported first-order convention before analysis. "
            "Soundform does not infer one from the channel count."
        )
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise AnalysisError(
            "First-order decoding requires exactly four channels. "
            "The file was not trimmed or padded to force a result."
        )
    if audio.shape[0] == 0:
        raise AnalysisError("There are no samples to decode.")

    if convention == "acn_sn3d":
        # File order W, Y, Z, X.
        pressure = audio[:, 0]
        left = audio[:, 1]
        up = audio[:, 2]
        forward = audio[:, 3]
    elif convention == "fuma":
        # File order W, X, Y, Z. FuMa W carries a 1/√2 factor.
        pressure = audio[:, 0] * np.sqrt(2.0)
        forward = audio[:, 1]
        left = audio[:, 2]
        up = audio[:, 3]
    else:
        # ACN order with N3D first-order scaling.
        pressure = audio[:, 0]
        left = audio[:, 1] / np.sqrt(3.0)
        up = audio[:, 2] / np.sqrt(3.0)
        forward = audio[:, 3] / np.sqrt(3.0)

    return np.column_stack((pressure, forward, left, up))
