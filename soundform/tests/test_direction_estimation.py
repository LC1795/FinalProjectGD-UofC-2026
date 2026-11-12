import numpy as np
import pytest
from scipy.signal import hilbert

from soundform.acoustics.sound_field import analyze_file
from soundform.acoustics.time_frequency import choose_stft_params
from soundform.audio.synthetic import encode_plane_wave
from soundform.tests.helpers import BANDS, angular_error_deg, plane_wave, signal, unit_vector


def _field(samples, sample_rate, convention="acn_sn3d", duration=None):
    seconds = duration if duration is not None else samples.shape[0] / sample_rate
    return analyze_file(samples, sample_rate, convention, 0.0, seconds, seconds, BANDS)


def _mean_direction(field):
    usable = np.isfinite(field.dominant_az) & np.isfinite(field.dominant_el)
    assert np.count_nonzero(usable) >= 1
    azimuth = np.deg2rad(field.dominant_az[usable])
    elevation = np.deg2rad(field.dominant_el[usable])
    vectors = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    )
    mean = vectors.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return mean


def _error_against(field, azimuth, elevation) -> float:
    mean = _mean_direction(field)
    target = unit_vector(azimuth, elevation)
    cosine = float(np.clip(np.dot(mean, target), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@pytest.mark.parametrize(
    ("azimuth", "elevation"),
    [
        (0.0, 0.0),
        (90.0, 0.0),
        (180.0, 0.0),
        (270.0, 0.0),
        (45.0, 0.0),
        (0.0, 30.0),
        (210.0, -20.0),
        (0.0, 80.0),
    ],
)
def test_known_directions_for_each_convention(azimuth, elevation):
    for convention in ("acn_sn3d", "fuma", "acn_n3d"):
        samples, sample_rate = plane_wave(azimuth, elevation, convention=convention, seconds=0.8, seed=3)
        field = _field(samples, sample_rate, convention, 0.8)
        assert _error_against(field, azimuth, elevation) < 10.0
        assert float(np.mean(field.bin_confidence)) > 0.7


def test_front_is_not_confused_with_back_or_left():
    front, sample_rate = plane_wave(0.0, 0.0, seconds=0.7, seed=4)
    back, _ = plane_wave(180.0, 0.0, seconds=0.7, seed=4)
    left, _ = plane_wave(90.0, 0.0, seconds=0.7, seed=4)
    front_error = _error_against(_field(front, sample_rate, duration=0.7), 0.0, 0.0)
    back_error = _error_against(_field(back, sample_rate, duration=0.7), 180.0, 0.0)
    left_error = _error_against(_field(left, sample_rate, duration=0.7), 90.0, 0.0)
    assert front_error < 8.0
    assert back_error < 8.0
    assert left_error < 8.0
    assert angular_error_deg(0, 0, 180, 0) > 170


def test_silence_and_near_silence_do_not_invent_a_direction():
    sample_rate = 16000
    silent = _field(np.zeros((sample_rate, 4)), sample_rate)
    assert silent.silent
    assert np.all(~np.isfinite(silent.dominant_az))
    assert float(np.max(silent.broadband)) == 0.0
    quiet = np.random.default_rng(1).normal(0.0, 1e-10, (sample_rate, 4))
    near = _field(quiet, sample_rate)
    assert near.silent
    assert "not measured sound pressure" in near.level_note or "not measured" in " ".join(near.warnings)


def test_frequency_bands_separate_low_and_high_sources():
    sample_rate = 16000
    count = sample_rate
    low = signal("tone", sample_rate, 1.0, 180.0, 1)
    high = signal("tone", sample_rate, 1.0, 4000.0, 1)
    mixed = encode_plane_wave(0.0, 0.0, low, "acn_sn3d") + encode_plane_wave(90.0, 0.0, high, "acn_sn3d")
    field = _field(mixed, sample_rate)
    weights = np.clip(np.cos(np.deg2rad(field.elevation_deg)), 0.0, None)

    def peak_azimuth(name: str) -> float:
        collapsed = (field.aggregate(name) * weights[:, None]).sum(axis=0)
        return float(field.azimuth_deg[int(np.argmax(collapsed))])

    assert angular_error_deg(peak_azimuth("low"), 0.0, 0.0, 0.0) < 15.0
    assert angular_error_deg(peak_azimuth("high"), 0.0, 90.0, 0.0) < 15.0
    assert field.aggregate("high").sum() > field.aggregate("low").sum() * 0.2


def test_sequential_and_moving_events_are_preserved():
    sample_rate = 16000
    first, _ = plane_wave(0.0, 0.0, seconds=1.0, seed=8)
    second, _ = plane_wave(90.0, 10.0, seconds=1.0, seed=9)
    field = _field(np.vstack([first, second]), sample_rate, duration=2.0)
    early = field.times < 0.55
    late = field.times > 1.45
    early_field_az = field.dominant_az.copy()
    assert np.any(np.isfinite(early_field_az[early]))
    assert np.any(np.isfinite(field.dominant_az[late]))
    early_mean = _mean_from_mask(field, early)
    late_mean = _mean_from_mask(field, late)
    assert float(np.degrees(np.arccos(np.clip(np.dot(early_mean, unit_vector(0, 0)), -1, 1)))) < 12
    assert float(np.degrees(np.arccos(np.clip(np.dot(late_mean, unit_vector(90, 10)), -1, 1)))) < 12


def _mean_from_mask(field, mask):
    azimuth = np.deg2rad(field.dominant_az[mask])
    elevation = np.deg2rad(field.dominant_el[mask])
    usable = np.isfinite(azimuth) & np.isfinite(elevation)
    vectors = np.column_stack(
        (
            np.cos(elevation[usable]) * np.cos(azimuth[usable]),
            np.cos(elevation[usable]) * np.sin(azimuth[usable]),
            np.sin(elevation[usable]),
        )
    )
    mean = vectors.mean(axis=0)
    return mean / np.linalg.norm(mean)


def test_interval_ignores_energy_outside_it():
    sample_rate = 16000
    quiet = np.zeros((sample_rate, 4))
    loud, _ = plane_wave(0.0, 0.0, seconds=0.8, seed=2)
    samples = np.vstack([quiet, loud])
    duration = samples.shape[0] / sample_rate
    field = analyze_file(samples, sample_rate, "acn_sn3d", 0.0, 0.6, duration, BANDS)
    assert field.silent or float(np.max(field.broadband)) < 1e-8


def test_diffuse_and_quadrature_fields_lose_confidence():
    sample_rate = 16000
    plane, _ = plane_wave(20.0, 0.0, seconds=0.8, seed=5)
    plane_field = _field(plane, sample_rate, duration=0.8)
    diffuse = np.random.default_rng(6).normal(0.0, 0.15, (int(sample_rate * 0.8), 4))
    diffuse_field = _field(diffuse, sample_rate, duration=0.8)
    assert float(np.mean(plane_field.bin_confidence)) > float(np.mean(diffuse_field.bin_confidence)) + 0.2
    assert float(np.mean(diffuse_field.agreement)) < 0.55

    tone = signal("noise", sample_rate, 0.8, 1000.0, 11)
    quadrature = np.imag(hilbert(tone))
    # ACN order W, Y, Z, X with X in quadrature against W.
    samples = np.column_stack([tone, np.zeros_like(tone), np.zeros_like(tone), quadrature])
    phase_field = _field(samples, sample_rate, duration=0.8)
    assert float(np.mean(phase_field.bin_confidence)) < 0.45


def test_heatmap_is_stable_and_windows_stay_inside_the_file():
    samples, sample_rate = plane_wave(15.0, 5.0, seconds=0.7, seed=12)
    first = _field(samples, sample_rate, duration=0.7)
    second = _field(samples, sample_rate, duration=0.7)
    assert np.allclose(first.broadband, second.broadband)
    assert np.all(np.diff(first.times) > 0)
    assert first.times[0] >= 0.0
    assert first.times[-1] <= 0.7
    nperseg, hop, coarsened = choose_stft_params(48000, 48000 * 90)
    assert coarsened
    assert hop >= nperseg // 2
