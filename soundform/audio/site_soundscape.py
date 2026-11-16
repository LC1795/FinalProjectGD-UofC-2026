"""Render a north-facing outdoor scene at a virtual Ambisonic microphone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from soundform.acoustics.higher_order import sn3d_basis
from soundform.audio.format_detection import append_axml_chunk
from soundform.audio.synthetic import _AMBIX_XML, write_wav
from soundform.geometry.design_space import EAR_HEIGHT_M


def write_site_soundscape(path: str | Path) -> dict[str, str]:
    path = Path(path)
    sample_rate, duration = 24000, 32
    time = np.arange(duration * sample_rate) / sample_rate
    rng = np.random.default_rng(21)
    channels = np.zeros((time.size, 16), dtype=np.float32)

    def noise(low, high):
        filtered = sosfilt(butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos"),
                           rng.standard_normal(time.size))
        return filtered / np.std(filtered)

    def add_source(wave, east, north, height, reference_distance):
        vertical = height - EAR_HEIGHT_M
        horizontal = np.hypot(east, north)
        gain = reference_distance / np.maximum(reference_distance, np.hypot(horizontal, vertical))
        bearing = np.degrees(np.arctan2(-np.asarray(east), north))
        elevation = np.degrees(np.arctan2(vertical, horizontal))
        if np.ndim(bearing) == 0:
            channels[:] += (wave * gain)[:, None] * sn3d_basis(bearing, elevation)
        else:
            for start in range(0, time.size, sample_rate):
                block = slice(start, start + sample_rate)
                basis = sn3d_basis(bearing[block], elevation[block])
                channels[block] += (wave[block] * gain[block])[:, None] * basis

    # Two lanes south of the listener. Each pass has its own speed and engine tone.
    vehicles = [(4, 12, -19), (7, -10, -23), (11, 15, -19), (16, -13, -23),
                (22, 11, -19), (26, -15, -23), (29, 13, -19)]
    add_source(0.028 * noise(100, 1350), 0, -25, 0.4, 25)
    for i, (closest, speed, north) in enumerate(vehicles):
        elapsed = time - closest
        engine_hz = 78 + 15 * (i % 4) - 4 * np.tanh(elapsed)
        engine = np.sin(2 * np.pi * np.cumsum(engine_hz) / sample_rate)
        vehicle = (0.066 * noise(170, 2300) + 0.038 * engine) * np.exp(-(elapsed / 3.8) ** 4)
        add_source(vehicle, speed * elapsed, north, 0.65, 19)

    # Two linked sections give the low rumble and wheel noise different trajectories.
    for delay, amplitude in ((0, 1.0), (1.7, 0.65)):
        elapsed = time - 17 - delay
        rumble = 0.145 * noise(35, 245) + 0.050 * noise(620, 2300)
        flange = np.sin(2 * np.pi * (1150 * elapsed + 7 * np.sin(2 * np.pi * 0.7 * elapsed)))
        rumble += 0.065 * flange * np.exp(-((elapsed - 1.2) / 1.35) ** 2)
        rail_joints = 0.55 + 0.45 * np.sin(2 * np.pi * 7.5 * elapsed) ** 4
        add_source(amplitude * rumble * rail_joints * np.exp(-(elapsed / 7.5) ** 4),
                   26, -8.5 * elapsed, 1.0, 26)

    # Water comes from several patches of shoreline and birds from two perches.
    for east, north in ((-15, 23), (-22, 21), (-18, 13)):
        water = noise(270, 2400) * (0.0012 + 0.0007 * np.sin(2 * np.pi * 0.43 * time + east))
        add_source(water, east, north, 0.0, 18)

    bird_groups = [(-17, 20, 7.5, [2, 5.5, 9, 13, 23, 28, 30.5]),
                   (-24, 15, 4.5, [3, 7.4, 10, 14, 20.5, 25, 29])]
    for east, north, height, onsets in bird_groups:
        birds = np.zeros(time.size)
        for onset in onsets:
            for chirp in range(3):
                start = int((onset + 0.115 * chirp) * sample_rate)
                local = np.arange(int(0.095 * sample_rate)) / sample_rate
                frequency = rng.uniform(1900, 2650)
                birds[start:start + local.size] += 0.015 * np.hanning(local.size) * np.sin(
                    2 * np.pi * (frequency * local + 4400 * local ** 2))
        add_source(birds, east, north, height, 23)

    # A low, distributed breeze keeps the scene from falling silent between events.
    for east, north in ((-18, 9), (14, 12), (-9, -14), (12, -8)):
        add_source(0.0018 * noise(200, 900), east, north, 2.4, 16)

    channels *= 0.85 / np.max(np.abs(channels))
    write_wav(path, channels, sample_rate, subtype="PCM_24")
    append_axml_chunk(path, _AMBIX_XML.replace("<order>1</order>", "<order>3</order>"))
    path.with_suffix(".scene.json").write_text(json.dumps({
        "duration_seconds": duration,
        "format": "Third-order ACN/SN3D (AmbiX)",
        "microphone": {"east_m": 0, "north_m": 0, "height_m": EAR_HEIGHT_M, "faces": "north"},
        "bearing_reference": "0 degrees north, +90 west, 180 south, -90 east",
        "traffic": [
            {"closest_s": closest, "eastward_speed_m_s": speed, "north_m": north}
            for closest, speed, north in vehicles
        ],
        "train": {"east_m": 26, "southward_speed_m_s": 8.5, "section_closest_s": [17, 18.7]},
        "pond": {"quadrant": "northwest", "bird_onsets_s": [onsets for _, _, _, onsets in bird_groups]},
    }, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "description": "32-second third-order scene facing north. Two-way traffic runs south of the microphone, "
                       "a southbound train passes on the east, and the northwest pond has quiet birds and water.",
    }

