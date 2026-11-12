"""Build a time-varying directional sound field from decoded FOA audio."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from soundform.acoustics.ambisonic_decoder import decode_foa
from soundform.acoustics.directional_energy import (
    N_AZIMUTH,
    N_ELEVATION,
    accumulate_direction_histogram,
    active_intensity,
    dominant_direction,
    grid_axes,
)
from soundform.acoustics.time_frequency import choose_stft_params, stft_wxyz, welch_relative_db
from soundform.audio.loader import amplitude_dbfs
from soundform.errors import AnalysisError

LEVEL_NOTE = (
    "Levels are relative to digital full scale (dBFS) or are normalized energy "
    "within this file. They are not measured sound pressure level in dB SPL. "
    "A first-order Ambisonic microphone does not determine source distance and "
    "does not reconstruct the room or street where the recording was made."
)

SILENCE_RMS = 1e-7


@dataclass
class SoundField:
    """Directional energy preserved over time, plus the labels needed to read it."""

    sample_rate: int
    convention: str
    interval: tuple[float, float]
    duration: float
    times: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    broadband: np.ndarray
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray
    cell_confidence: np.ndarray
    bin_confidence: np.ndarray
    agreement: np.ndarray
    dominant_az: np.ndarray
    dominant_el: np.ndarray
    frame_level_db: np.ndarray
    freq_hz: np.ndarray
    spectrum_db_rel: np.ndarray
    rms_dbfs: float
    peak_dbfs: float
    silent: bool
    warnings: list[str] = field(default_factory=list)
    band_limits: dict = field(default_factory=dict)
    stft_hop: int = 0
    stft_nperseg: int = 0
    level_note: str = LEVEL_NOTE

    def band(self, name: str) -> np.ndarray:
        if name == "broadband":
            return self.broadband
        if name == "low":
            return self.low
        if name == "mid":
            return self.mid
        if name == "high":
            return self.high
        raise KeyError(name)

    def aggregate(self, name: str) -> np.ndarray:
        data = self.band(name)
        if data.size == 0 or data.shape[0] == 0:
            return np.zeros(data.shape[1:], dtype=np.float64)
        return data.mean(axis=0)

    def frame_index_at(self, time_s: float) -> int:
        if self.times.size == 0:
            return 0
        return int(np.clip(np.searchsorted(self.times, time_s), 0, len(self.times) - 1))

    def display_map(self, name: str, time_s: float | None, smoothing: float) -> np.ndarray:
        """Relative energy for the plots. Not a calibrated sound pressure level.

        Smoothing happens before normalisation, so a wide kernel cannot crush
        the colour scale toward black. The scale is the strongest direction of
        the smoothed interval average. A quieter moment stays below that peak.
        """
        from soundform.acoustics.directional_energy import smooth_azimuth_elevation

        sigma = max(0.0, float(smoothing)) * 4.0
        band = self.band(name)
        if band.size == 0 or float(np.max(band)) <= 1e-20:
            return np.zeros_like(self.aggregate(name), dtype=np.float64)
        reference = smooth_azimuth_elevation(np.asarray(self.aggregate(name), dtype=np.float64), sigma)
        peak = float(np.max(reference))
        if time_s is None:
            shown = reference
        else:
            frame = np.asarray(band[self.frame_index_at(time_s)], dtype=np.float64)
            shown = smooth_azimuth_elevation(frame, sigma)
        if peak <= 1e-20:
            return np.zeros_like(shown, dtype=np.float64)
        return np.asarray(shown, dtype=np.float64) / peak

    def to_npz(self, path: str) -> None:
        meta = {
            "sample_rate": self.sample_rate,
            "convention": self.convention,
            "interval": list(self.interval),
            "duration": self.duration,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "silent": self.silent,
            "warnings": self.warnings,
            "band_limits": {key: list(value) for key, value in self.band_limits.items()},
            "stft_hop": self.stft_hop,
            "stft_nperseg": self.stft_nperseg,
            "level_note": self.level_note,
        }
        np.savez_compressed(
            path,
            times=self.times,
            azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg,
            broadband=self.broadband,
            low=self.low,
            mid=self.mid,
            high=self.high,
            cell_confidence=self.cell_confidence,
            bin_confidence=self.bin_confidence,
            agreement=self.agreement,
            dominant_az=self.dominant_az,
            dominant_el=self.dominant_el,
            frame_level_db=self.frame_level_db,
            freq_hz=self.freq_hz,
            spectrum_db_rel=self.spectrum_db_rel,
            meta=np.array(json.dumps(meta)),
        )

    @classmethod
    def from_npz(cls, path: str) -> "SoundField":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                sample_rate=int(meta["sample_rate"]),
                convention=str(meta["convention"]),
                interval=(float(meta["interval"][0]), float(meta["interval"][1])),
                duration=float(meta["duration"]),
                times=data["times"],
                azimuth_deg=data["azimuth_deg"],
                elevation_deg=data["elevation_deg"],
                broadband=data["broadband"],
                low=data["low"],
                mid=data["mid"],
                high=data["high"],
                cell_confidence=data["cell_confidence"],
                bin_confidence=data["bin_confidence"],
                agreement=data["agreement"],
                dominant_az=data["dominant_az"],
                dominant_el=data["dominant_el"],
                frame_level_db=data["frame_level_db"],
                freq_hz=data["freq_hz"],
                spectrum_db_rel=data["spectrum_db_rel"],
                rms_dbfs=float(meta["rms_dbfs"]),
                peak_dbfs=float(meta["peak_dbfs"]),
                silent=bool(meta["silent"]),
                warnings=list(meta["warnings"]),
                band_limits={key: (float(value[0]), float(value[1])) for key, value in meta["band_limits"].items()},
                stft_hop=int(meta["stft_hop"]),
                stft_nperseg=int(meta["stft_nperseg"]),
                level_note=str(meta["level_note"]),
            )


def _frequency_mask(frequencies: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    if high <= low:
        return np.zeros(frequencies.shape, dtype=bool)
    return (frequencies >= low) & (frequencies < high)


def _prepare_limits(
    sample_rate: int,
    requested: dict[str, tuple[float, float]],
    warnings: list[str],
) -> dict[str, tuple[float, float]]:
    nyquist = sample_rate / 2.0
    prepared: dict[str, tuple[float, float]] = {}
    for name in ("low", "mid", "high"):
        if name not in requested:
            raise AnalysisError(f"Missing frequency limits for the {name} band.")
        low, high = requested[name]
        if high <= low:
            raise AnalysisError(f"The {name} band maximum must be above its minimum.")
        clamped_high = min(float(high), nyquist * 0.999)
        clamped_low = max(0.0, float(low))
        if clamped_high <= clamped_low:
            warnings.append(
                f"The {name} band sits at or above Nyquist ({nyquist:.0f} Hz) and is empty for this file."
            )
            prepared[name] = (clamped_low, clamped_low)
        else:
            if clamped_high < float(high) - 1.0:
                warnings.append(
                    f"The {name} band was limited from {high:.0f} Hz to {clamped_high:.0f} Hz by the sample rate."
                )
            prepared[name] = (clamped_low, clamped_high)
    return prepared


def _empty_maps(n_frames: int, n_elevation: int, n_azimuth: int) -> dict[str, np.ndarray]:
    shape = (n_frames, n_elevation, n_azimuth)
    return {name: np.zeros(shape, dtype=np.float32) for name in ("broadband", "low", "mid", "high")}


def analyze_file(
    samples: np.ndarray,
    sample_rate: int,
    convention: str,
    start_s: float,
    end_s: float,
    file_duration: float,
    band_limits: dict[str, tuple[float, float]],
) -> SoundField:
    """Decode `samples` with `convention` and estimate a time-varying sound field."""
    decoded = decode_foa(samples, convention)
    if end_s <= start_s:
        raise AnalysisError("The analysis interval end must be later than the start.")
    if start_s < -1e-6 or end_s > file_duration + 0.05:
        raise AnalysisError("The analysis interval extends outside the recording.")

    start_index = max(0, int(np.floor(start_s * sample_rate)))
    end_index = min(decoded.shape[0], int(np.ceil(end_s * sample_rate)))
    segment = decoded[start_index:end_index]
    if segment.shape[0] < int(0.25 * sample_rate):
        raise AnalysisError("Select an analysis interval of at least 0.25 seconds.")

    warnings = [
        LEVEL_NOTE,
        "Each time-frequency tile contributes one arrival direction. Sources that overlap "
        "in the same tile are not separated. The map is a first-order estimate, not a "
        "high-resolution image of the sound field.",
        "Azimuth 0° is microphone forward and positive azimuth is toward microphone left. "
        "Elevation is positive upward. These are microphone axes, not a compass bearing "
        "and not a site plan.",
    ]
    pressure = segment[:, 0]
    rms = float(np.sqrt(np.mean(np.square(pressure)))) if pressure.size else 0.0
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    rms_db = amplitude_dbfs(rms)
    peak_db = amplitude_dbfs(peak)
    used_limits = _prepare_limits(sample_rate, band_limits, warnings)
    azimuth, elevation, edges = grid_axes()

    nperseg, hop, coarsened = choose_stft_params(sample_rate, segment.shape[0])
    if coarsened:
        warnings.append(
            "The recording is long, so the analysis hop was coarsened to keep the "
            "time-varying map responsive. Short events narrower than one hop can be averaged."
        )

    if rms < SILENCE_RMS or peak < SILENCE_RMS:
        warnings.append(
            "This interval is effectively silent. Direction is undefined, and no arrival has been invented."
        )
        times = np.array([start_s + 0.5 * (end_s - start_s)], dtype=np.float64)
        maps = _empty_maps(1, len(elevation), len(azimuth))
        return SoundField(
            sample_rate=sample_rate,
            convention=convention,
            interval=(float(start_s), float(end_s)),
            duration=float(file_duration),
            times=times,
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            broadband=maps["broadband"],
            low=maps["low"],
            mid=maps["mid"],
            high=maps["high"],
            cell_confidence=np.zeros((1, len(elevation), len(azimuth)), dtype=np.float32),
            bin_confidence=np.zeros(1, dtype=np.float32),
            agreement=np.zeros(1, dtype=np.float32),
            dominant_az=np.array([np.nan]),
            dominant_el=np.array([np.nan]),
            frame_level_db=np.array([-120.0], dtype=np.float32),
            freq_hz=np.array([0.0]),
            spectrum_db_rel=np.array([-120.0]),
            rms_dbfs=rms_db,
            peak_dbfs=peak_db,
            silent=True,
            warnings=warnings,
            band_limits=used_limits,
            stft_hop=hop,
            stft_nperseg=nperseg,
        )

    frequencies, times, spectra = stft_wxyz(segment, sample_rate, nperseg, hop)
    times = times + (start_index / sample_rate)
    intensity = active_intensity(spectra[0], spectra[1], spectra[2], spectra[3])
    intensity_x, intensity_y, intensity_z, energy, confidence = intensity
    # Sub-bass bins are poorly localised by a short Fourier window and are
    # excluded from the directional estimate. They still contribute to RMS.
    audible = frequencies >= 20.0
    if np.any(audible) and not np.all(audible):
        keep = audible[:, None]
        intensity_x = np.where(keep, intensity_x, 0.0)
        intensity_y = np.where(keep, intensity_y, 0.0)
        intensity_z = np.where(keep, intensity_z, 0.0)
        energy = np.where(keep, energy, 0.0)
        confidence = np.where(keep, confidence, 0.0)

    broadband_limits = (20.0, sample_rate / 2.0 * 0.999)
    masks = {
        "broadband": _frequency_mask(frequencies, broadband_limits),
        "low": _frequency_mask(frequencies, used_limits["low"]),
        "mid": _frequency_mask(frequencies, used_limits["mid"]),
        "high": _frequency_mask(frequencies, used_limits["high"]),
    }
    maps: dict[str, np.ndarray] = {}
    cell_confidence = None
    for name, mask in masks.items():
        histogram, confidence_acc = accumulate_direction_histogram(
            intensity_x,
            intensity_y,
            intensity_z,
            energy,
            confidence,
            mask,
            len(azimuth),
            len(elevation),
            edges,
        )
        maps[name] = histogram.astype(np.float32)
        if name == "broadband":
            cell_confidence = np.divide(
                confidence_acc,
                histogram,
                out=np.zeros_like(histogram),
                where=histogram > 1e-20,
            ).astype(np.float32)
    if cell_confidence is None:
        cell_confidence = np.zeros_like(maps["broadband"])

    band_confidence = (confidence * energy).sum(axis=0) / (energy.sum(axis=0) + 1e-20)
    dominant_az, dominant_el, agreement = dominant_direction(intensity_x, intensity_y, intensity_z)
    frame_power = energy.sum(axis=0)
    peak_power = float(np.max(frame_power)) if frame_power.size else 0.0
    if peak_power <= 1e-20:
        frame_level = np.full(frame_power.shape, -120.0)
    else:
        frame_level = 10.0 * np.log10(np.maximum(frame_power, 1e-20) / peak_power)

    mean_confidence = float(np.average(band_confidence, weights=np.maximum(frame_power, 1e-20)))
    mean_agreement = float(np.average(agreement, weights=np.maximum(frame_power, 1e-20)))
    if mean_confidence < 0.35:
        warnings.append(
            "Directional confidence is low. Energy is spread over the sphere because the "
            "field is diffuse, quiet, or out of phase, rather than pinned to one arrival."
        )
    if mean_agreement < 0.35 and mean_confidence >= 0.35:
        warnings.append(
            "Several directions are active together. A single dominant angle would fall "
            "between them, so the timeline withholds that angle where the bins disagree. "
            "Use the directional map, which keeps the separate arrivals."
        )

    spectrum_f, spectrum_db = welch_relative_db(pressure, sample_rate, nperseg)
    return SoundField(
        sample_rate=sample_rate,
        convention=convention,
        interval=(float(start_s), float(end_s)),
        duration=float(file_duration),
        times=times.astype(np.float64),
        azimuth_deg=azimuth.astype(np.float64),
        elevation_deg=elevation.astype(np.float64),
        broadband=maps["broadband"],
        low=maps["low"],
        mid=maps["mid"],
        high=maps["high"],
        cell_confidence=cell_confidence,
        bin_confidence=band_confidence.astype(np.float32),
        agreement=agreement.astype(np.float32),
        dominant_az=dominant_az,
        dominant_el=dominant_el,
        frame_level_db=frame_level.astype(np.float32),
        freq_hz=spectrum_f,
        spectrum_db_rel=spectrum_db,
        rms_dbfs=rms_db,
        peak_dbfs=peak_db,
        silent=False,
        warnings=warnings,
        band_limits=used_limits,
        stft_hop=hop,
        stft_nperseg=nperseg,
    )
