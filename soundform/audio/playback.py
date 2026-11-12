"""Stereo monitoring of a decoded Ambisonic bed.

The downmix is only for headphones or speakers. Direction estimates never use it.
Y is the microphone's left axis, so left = W + Y and right = W − Y, with a small
amount of X so a frontal source is not only in the sum.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - import failure is reported when playback starts
    sd = None


def stereo_monitor_downmix(wxyz: np.ndarray) -> np.ndarray:
    """Fold SN3D W, X, Y, Z into a stereo preview at a safe peak level."""
    audio = np.asarray(wxyz, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError("The monitor downmix expects four decoded channels.")
    pressure, forward, left, _up = audio.T
    stereo = np.column_stack(
        (
            pressure + left + 0.25 * forward,
            pressure - left + 0.25 * forward,
        )
    )
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1e-9:
        stereo *= 0.4 / peak
    return stereo.astype(np.float32)


def average_channel_downmix(samples: np.ndarray) -> np.ndarray:
    """Non-spatial preview used when no Ambisonic convention has been selected."""
    audio = np.asarray(samples, dtype=np.float64)
    mono = np.mean(audio, axis=1)
    stereo = np.column_stack((mono, mono))
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1e-9:
        stereo *= 0.4 / peak
    return stereo.astype(np.float32)


class MonitorPlayer:
    """Plays a stereo buffer from a movable frame cursor.

    The audio callback only advances the frame counter. The interface polls
    `position_seconds` from a timer and never touches Qt from the callback.
    """

    def __init__(self) -> None:
        self.samples: np.ndarray | None = None
        self.sample_rate = 48000
        self._frame = 0
        self._stream = None
        self._lock = threading.Lock()
        self._playing = False

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def duration(self) -> float:
        if self.samples is None or self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples) / self.sample_rate)

    @property
    def position_seconds(self) -> float:
        with self._lock:
            frame = self._frame
        if self.sample_rate <= 0:
            return 0.0
        return frame / self.sample_rate

    def load(self, stereo: np.ndarray, sample_rate: int) -> None:
        self.stop()
        audio = np.asarray(stereo, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Monitor playback expects a stereo buffer.")
        with self._lock:
            self.samples = np.ascontiguousarray(audio)
            self.sample_rate = int(sample_rate)
            self._frame = 0

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self.samples = None
            self._frame = 0

    def play(self) -> None:
        if sd is None:
            raise RuntimeError(
                "Playback needs the sounddevice package and a working output device."
            )
        if self.samples is None:
            raise RuntimeError("Import a recording before playing.")
        if self._playing:
            return
        with self._lock:
            if self._frame >= len(self.samples):
                self._frame = 0

        def callback(outdata, frames, _time_info, _status):
            with self._lock:
                if self.samples is None:
                    outdata.fill(0)
                    raise sd.CallbackStop()
                start = self._frame
                end = start + frames
                chunk = self.samples[start:end]
                count = len(chunk)
                if count:
                    outdata[:count] = chunk
                if count < frames:
                    outdata[count:] = 0
                    self._frame = len(self.samples)
                    raise sd.CallbackStop()
                self._frame = end

        def finished():
            self._playing = False

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
            callback=callback,
            finished_callback=finished,
        )
        self._stream.start()
        self._playing = True

    def pause(self) -> None:
        self._close_stream()
        self._playing = False

    def stop(self) -> None:
        self._close_stream()
        self._playing = False
        with self._lock:
            self._frame = 0

    def seek(self, seconds: float) -> None:
        if self.samples is None or self.sample_rate <= 0:
            return
        frame = int(np.clip(seconds, 0.0, self.duration) * self.sample_rate)
        with self._lock:
            self._frame = min(frame, len(self.samples))

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
