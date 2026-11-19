"""Render a north-facing downtown park as synthetic third-order Ambisonics."""

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
    sample_rate, duration = 24000, 45
    time = np.arange(duration * sample_rate) / sample_rate
    rng = np.random.default_rng(74)
    channels = np.zeros((time.size, 16), dtype=np.float32)
    events = []

    def noise(low, high):
        wave = sosfilt(butter(2, [low, high], btype="bandpass",
                              fs=sample_rate, output="sos"), rng.standard_normal(time.size))
        return wave / max(float(np.std(wave)), 1e-12)

    def add_source(wave, east, north, height=0.7, reference_distance=25, reflection=0.08):
        east = np.broadcast_to(east, time.shape)
        north = np.broadcast_to(north, time.shape)
        vertical = np.broadcast_to(np.asarray(height) - EAR_HEIGHT_M, time.shape)
        horizontal = np.hypot(east, north)
        distance = np.hypot(horizontal, vertical)
        gain = reference_distance / np.maximum(reference_distance, distance)
        # Greater distance removes some fine detail without changing the source bearing.
        wave = np.asarray(wave) * gain
        for start in range(0, time.size, sample_rate):
            block = slice(start, start + sample_rate)
            bearing = np.degrees(np.arctan2(-east[block], north[block]))
            elevation = np.degrees(np.arctan2(vertical[block], horizontal[block]))
            channels[block] += wave[block, None] * sn3d_basis(bearing, elevation)
        # Sparse, quiet echoes give downtown facades a spatial presence.
        if reflection:
            reflected = sosfilt(butter(1, 1600, fs=sample_rate, output="sos"), wave)
            delay = int(0.18 * sample_rate)
            reflected = np.r_[np.zeros(delay), reflected[:-delay]] * reflection
            for bearing in (75, 205, 290):
                channels[:] += reflected[:, None] * sn3d_basis(bearing, 7) / 3

    def pulse(onset, seconds, frequency, amplitude):
        local = time - onset
        envelope = np.where((local >= 0) & (local < seconds),
                            np.exp(-np.maximum(local, 0) * 9 / seconds), 0)
        return amplitude * envelope * np.sin(2 * np.pi * frequency * local)

    # Continuous two-way roads beyond the south and west boundaries.
    add_source(0.022 * noise(65, 1800), -20, -67, reference_distance=40)
    add_source(0.025 * noise(70, 1900), -66, -16, reference_distance=40)
    for index, closest in enumerate((2, 6, 10, 15, 19, 24, 29, 34, 39, 43)):
        speed = (10.5 if index % 2 == 0 else -12)
        north = -65 if speed > 0 else -69
        elapsed = time - closest
        frequency = (68 + 11 * (index % 5)) - 3 * np.tanh(elapsed / 3)
        engine = np.sin(2 * np.pi * np.cumsum(frequency) / sample_rate)
        body = (0.065 * noise(95, 2200) + 0.033 * engine) * np.exp(-(elapsed / 7) ** 4)
        add_source(body, speed * elapsed, north, reference_distance=40)
        events.append({"kind": "south road vehicle", "closest_s": closest,
                       "eastward_speed_m_s": speed, "north_m": north})
    for index, closest in enumerate((4, 9, 14, 20, 26, 31, 37, 42)):
        speed = 9 if index % 2 == 0 else -11
        east = -65 if speed > 0 else -69
        elapsed = time - closest
        engine = np.sin(2 * np.pi * (82 + 13 * (index % 3)) * time)
        body = (0.072 * noise(90, 2000) + 0.035 * engine) * np.exp(-(elapsed / 7) ** 4)
        add_source(body, east, speed * elapsed, reference_distance=40)
        events.append({"kind": "west road vehicle", "closest_s": closest,
                       "northward_speed_m_s": speed, "east_m": east})

    # Two horn presses from a moving car on the west road.
    horn = np.zeros(time.size)
    for onset, seconds in ((16.3, 0.35), (16.95, 0.8)):
        phase = time - onset
        gate = np.clip(phase / 0.025, 0, 1) * np.clip((seconds - phase) / 0.045, 0, 1)
        tone = sum(np.sin(2 * np.pi * hz * phase) / harmonic
                   for hz, harmonic in ((392, 1), (494, 1), (784, 3), (988, 3)))
        horn += 0.12 * gate * tone
    add_source(horn, -65, 9 * (time - 17), height=0.8, reference_distance=40, reflection=0.17)
    events.append({"kind": "car horn", "onsets_s": [16.3, 16.95], "road": "west"})

    # A three-car commuter train moves west to east, 25 m north of the deck.
    for carriage, closest in enumerate((22, 23.6, 25.2)):
        elapsed = time - closest
        joints = 0.62 + 0.38 * np.sin(2 * np.pi * 5.7 * elapsed) ** 8
        rumble = 0.13 * noise(35, 300) + 0.042 * noise(450, 2800)
        whine = 0.022 * np.sin(2 * np.pi * (680 * time + 6 * np.sin(2 * np.pi * 0.45 * time)))
        train = (rumble + whine) * joints * np.exp(-(elapsed / 8) ** 4)
        add_source(train, 13 * elapsed, 25, height=1.1, reference_distance=25, reflection=0.13)
        events.append({"kind": "train carriage", "closest_s": closest,
                       "eastward_speed_m_s": 13, "north_m": 25})

    # Basketball in the northwest quadrant. Irregular dribbles, impacts and shoe squeaks.
    basketball = np.zeros(time.size)
    for onset in (1.8, 2.5, 3.2, 4.0, 7.7, 8.3, 9.0, 12.4, 13.1, 18.5, 19.2,
                  27.5, 28.2, 28.8, 32.5, 33.3, 38.0, 38.7, 39.5, 42.5):
        basketball += pulse(onset, 0.35, rng.uniform(82, 115), 0.07)
        local = time - onset
        basketball += 0.022 * noise(350, 2700) * np.where(
            (local >= 0) & (local < 0.07), np.exp(-np.maximum(local, 0) * 90), 0)
    for onset in (5.0, 11.4, 29.6, 40.8):
        local = time - onset
        basketball += 0.012 * np.exp(-(local / 0.09) ** 2) * np.sin(
            2 * np.pi * (1250 * local + 280 * local ** 2))
    add_source(basketball, -37, -3, height=0.2, reference_distance=20, reflection=0.06)

    # A few walkers along north-south concrete paths, without conversations.
    for index, (closest, east, speed) in enumerate(((8, -6, -1.25), (31, 9, 1.1))):
        steps = np.zeros(time.size)
        for onset in np.arange(0.4 + index * 0.31, duration, 0.58 + 0.03 * index):
            local = time - onset
            step = np.where((local >= 0) & (local < 0.13), np.exp(-np.maximum(local, 0) * 45), 0)
            steps += (0.008 * noise(150, 2400) + pulse(onset, 0.13, 170, 0.008)) * step
        add_source(steps, east, speed * (time - closest), height=0.1, reference_distance=6)

    # The southeastern green quadrant is calmer, with a raised hill and occasional birds.
    add_source(0.004 * noise(250, 1600) * (0.8 + 0.2 * np.sin(0.6 * time)),
               5, -31, height=3, reference_distance=20, reflection=0)
    for onset in (5.8, 12.6, 30.5, 36.2):
        local = time - onset
        chirp = 0.008 * np.exp(-(local / 0.085) ** 2) * np.sin(
            2 * np.pi * (2300 * local + 1700 * local ** 2))
        add_source(chirp, 8, -34, height=5, reference_distance=20, reflection=0)

    # Quiet service-building fans in the southwest and tower ventilation to the east.
    for east, north in ((-44, -38), (-32, -45)):
        add_source(0.006 * noise(90, 600), east, north, height=2.2, reference_distance=20)
    tower = 0.011 * noise(70, 700) + 0.003 * np.sin(2 * np.pi * 120 * time)
    add_source(tower, 25, 0, height=8, reference_distance=25)
    for east, north in ((-90, 65), (65, 75), (70, -100), (-100, -90)):
        add_source(0.005 * noise(90, 1700), east, north, height=5, reference_distance=45, reflection=0)
    for east, north in ((-15, 14), (14, 9), (-7, -20)):
        add_source(0.002 * noise(200, 1200), east, north, height=3, reference_distance=15, reflection=0)

    fade = np.minimum(np.clip(time / 0.12, 0, 1), np.clip((duration - time) / 0.2, 0, 1))
    channels *= fade[:, None]
    channels *= 0.88 / max(float(np.max(np.abs(channels))), 1e-12)
    write_wav(path, channels, sample_rate, subtype="PCM_24")
    append_axml_chunk(path, _AMBIX_XML.replace("<order>1</order>", "<order>3</order>"))
    scene = {
        "duration_seconds": duration, "format": "Third-order ACN/SN3D (AmbiX)",
        "synthetic": True,
        "microphone": {"east_m": 0, "north_m": 0, "height_m": EAR_HEIGHT_M,
                       "faces": "north", "location": "deck in northeast park quadrant"},
        "park_bounds_m": {"east": 20, "west": -61, "north": 20, "south": -61},
        "basketball_court": {"quadrant": "northwest", "east_m": -37, "north_m": -3},
        "services": {"quadrant": "southwest", "buildings_xy_m": [[-44,-38],[-32,-45]]},
        "hill": {"quadrant": "southeast", "east_m": 5, "north_m": -31, "source_height_m": 3},
        "tower": {"east_m": 25, "north_m": 0, "storeys": 60},
        "west_road_distance_m_assumed": 65,
        "events": events,
        "notes": "Procedural sound textures and sparse facade echoes. Distances guide source bearings "
                 "and level attenuation. Geometry-based occlusion and calibrated SPL are not simulated.",
    }
    path.with_suffix(".scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return {"path": str(path), "description":
            "45-second synthetic downtown park scene, facing north from the northeast deck. "
            "West-to-east train to the north, two-way roads south and west, car horn, northwest "
            "basketball, sparse footsteps, quiet southeast greenery and downtown ventilation."}
