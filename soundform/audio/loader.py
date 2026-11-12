"""Read and validate WAV recordings.

Samples are returned in the file's own channel order. Decoding happens later,
and only after a convention is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from soundform.audio.format_detection import FormatDetection, detect_spatial_format
from soundform.errors import AudioLoadError

_BIT_DEPTH = {
    "PCM_U8": "8-bit PCM",
    "PCM_S8": "8-bit PCM",
    "PCM_16": "16-bit PCM",
    "PCM_24": "24-bit PCM",
    "PCM_32": "32-bit PCM",
    "FLOAT": "32-bit float",
    "DOUBLE": "64-bit float",
}

_MIN_SAMPLE_RATE = 8000
_MAX_SAMPLE_RATE = 384000


@dataclass
class ChannelStats:
    index: int
    rms_dbfs: float
    peak_dbfs: float


@dataclass
class AudioFile:
    path: str
    samples: np.ndarray
    sample_rate: int
    channels: int
    frames: int
    duration: float
    subtype: str
    bit_depth: str
    detection: FormatDetection
    channel_stats: list[ChannelStats] = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name


def amplitude_dbfs(value: float) -> float:
    """20·log10 of a linear amplitude. Values at or below the floor read as −120 dBFS."""
    if not np.isfinite(value) or value <= 1e-12:
        return -120.0
    return float(20.0 * np.log10(value))


def channel_levels(samples: np.ndarray) -> list[ChannelStats]:
    stats: list[ChannelStats] = []
    for index in range(samples.shape[1]):
        column = samples[:, index].astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(column)))) if column.size else 0.0
        peak = float(np.max(np.abs(column))) if column.size else 0.0
        stats.append(ChannelStats(index, amplitude_dbfs(rms), amplitude_dbfs(peak)))
    return stats


def _bit_depth_label(subtype: str | None) -> str:
    if not subtype:
        return "not available"
    return _BIT_DEPTH.get(subtype, subtype)


def load_audio(path: str | Path) -> AudioFile:
    """Load a WAV file and attach a spatial-format classification.

    Mono, stereo, and other unsupported layouts are still returned so the
    interface can show their metadata. Their detection status blocks analysis.
    Corrupt files, empty files, and unreadable codecs raise AudioLoadError.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioLoadError("That file does not exist.")
    if file_path.suffix.lower() not in {".wav", ".wave"}:
        raise AudioLoadError(
            "Unsupported format. Import a WAV recording. Soundform does not "
            "read MP3, FLAC, or other containers for spatial analysis."
        )
    if file_path.stat().st_size < 44:
        raise AudioLoadError("The file is empty or too small to be a WAV recording.")

    try:
        info = sf.info(str(file_path))
    except Exception as exc:
        raise AudioLoadError(
            "The file could not be read. It may be corrupt or use a codec "
            f"this installer of libsndfile does not support. ({exc})"
        ) from exc

    if info.frames is None or info.frames <= 0:
        raise AudioLoadError("The file contains no audio frames.")
    sample_rate = int(info.samplerate or 0)
    if sample_rate < _MIN_SAMPLE_RATE or sample_rate > _MAX_SAMPLE_RATE:
        raise AudioLoadError(
            f"Sample rate {sample_rate} Hz is not supported. Use a PCM or float "
            f"WAV between {_MIN_SAMPLE_RATE} and {_MAX_SAMPLE_RATE} Hz."
        )
    if info.channels is None or info.channels < 1:
        raise AudioLoadError("The file does not declare any audio channels.")

    try:
        samples, read_rate = sf.read(str(file_path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioLoadError(
            "The audio frames could not be decoded. The file may be corrupt "
            f"or use an unsupported WAV codec. ({exc})"
        ) from exc

    if samples.size == 0 or samples.shape[0] == 0:
        raise AudioLoadError("The file contains no audio frames.")
    if not np.isfinite(samples).all():
        raise AudioLoadError("The file contains NaN or infinite sample values.")
    if int(read_rate) != sample_rate:
        sample_rate = int(read_rate)

    detection = detect_spatial_format(file_path, int(samples.shape[1]))
    return AudioFile(
        path=str(file_path.resolve()),
        samples=samples,
        sample_rate=sample_rate,
        channels=int(samples.shape[1]),
        frames=int(samples.shape[0]),
        duration=float(samples.shape[0] / sample_rate),
        subtype=str(info.subtype or ""),
        bit_depth=_bit_depth_label(info.subtype),
        detection=detection,
        channel_stats=channel_levels(samples),
    )
