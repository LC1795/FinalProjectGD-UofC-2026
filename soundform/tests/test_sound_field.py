import numpy as np
import pytest

from soundform.acoustics.response import compute_response
from soundform.acoustics.sound_field import SoundField
from soundform.geometry.design_space import design_xy_to_sound_azimuth, sound_az_el_to_design_vector
from soundform.tests.helpers import blob_field


def test_response_peaks_on_the_blob_and_survives_zero_weights():
    field = blob_field(30.0, 0.0)
    response = compute_response(
        field,
        {
            "broadband": 0.4,
            "low": 0.1,
            "mid": 0.1,
            "high": 0.0,
            "persistence": 0.2,
            "transient": 0.0,
            "confidence": 0.2,
        },
    )
    assert np.all(np.isfinite(response.score))
    peak = int(np.argmax(response.azimuth_score))
    assert abs(((response.azimuth_deg[peak] - 30.0 + 180) % 360) - 180) < 12.0
    flat = compute_response(field, {key: 0.0 for key in ("broadband", "low", "mid", "high", "persistence", "transient", "confidence")})
    assert np.all(flat.score == 0.0)
    assert flat.warnings


def test_persistence_prefers_a_steady_direction():
    field = blob_field(0.0, transient_azimuth=180.0, n_time=8)
    response = compute_response(
        field,
        {
            "broadband": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "persistence": 1.0,
            "transient": 0.0,
            "confidence": 0.0,
        },
    )
    front = response.azimuth_score[np.argmin(np.abs(response.azimuth_deg - 2.5))]
    back_index = int(np.argmin(np.abs(((response.azimuth_deg - 180.0 + 180) % 360) - 180)))
    back = response.azimuth_score[back_index]
    assert front > back


def test_transient_weight_can_prefer_the_spike():
    field = blob_field(0.0, transient_azimuth=180.0, n_time=8)
    response = compute_response(
        field,
        {
            "broadband": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "persistence": 0.0,
            "transient": 1.0,
            "confidence": 0.0,
        },
    )
    front_index = int(np.argmin(np.abs(response.azimuth_deg - 2.5)))
    back_index = int(np.argmin(np.abs(((response.azimuth_deg - 180.0 + 180) % 360) - 180)))
    assert response.transient[0, back_index] >= response.transient[0, front_index]


def test_display_scale_uses_the_interval_peak():
    field = blob_field()
    field.broadband = field.broadband.copy()
    field.broadband[-1] *= 0.1
    quiet = field.display_map("broadband", time_s=float(field.times[-1]), smoothing=0.0)
    full = field.display_map("broadband", time_s=None, smoothing=0.0)
    assert float(np.max(quiet)) < float(np.max(full))


def test_rotation_moves_microphone_forward_counter_clockwise():
    vector = sound_az_el_to_design_vector(0.0, 0.0, 0.0)
    assert vector[1] > 0.9
    assert abs(vector[0]) < 0.1
    turned = sound_az_el_to_design_vector(0.0, 0.0, 90.0)
    assert turned[0] < -0.9
    heard = design_xy_to_sound_azimuth(-1.0, 0.0, 90.0)
    assert abs(((heard + 180) % 360) - 180) < 1e-6


def test_sound_field_roundtrip(tmp_path):
    field = blob_field(12.0, 4.0)
    path = tmp_path / "field.npz"
    field.to_npz(str(path))
    loaded = SoundField.from_npz(str(path))
    assert loaded.convention == "acn_sn3d"
    assert np.allclose(loaded.broadband, field.broadband)
    assert "not measured" in loaded.level_note
    assert "sound pressure level" in loaded.level_note
    assert np.isfinite(loaded.rms_dbfs)


def test_zero_energy_response_has_no_nan():
    field = blob_field()
    field.broadband = np.zeros_like(field.broadband)
    field.low = np.zeros_like(field.low)
    field.mid = np.zeros_like(field.mid)
    field.high = np.zeros_like(field.high)
    field.cell_confidence = np.zeros_like(field.cell_confidence)
    field.silent = True
    response = compute_response(field, {"broadband": 1, "low": 1, "mid": 1, "high": 1, "persistence": 1, "transient": 1, "confidence": 1})
    assert np.all(response.score == 0)
    assert np.all(np.isfinite(response.azimuth_score))
