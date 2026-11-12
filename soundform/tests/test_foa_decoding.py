import numpy as np

from soundform.acoustics.ambisonic_decoder import decode_foa
from soundform.audio.synthetic import encode_plane_wave
from soundform.errors import AnalysisError
import pytest


def test_acn_sn3d_channel_order_and_scaling():
    signal = np.sin(np.linspace(0, 12, 500))
    encoded = encode_plane_wave(0.0, 0.0, signal, "acn_sn3d")
    assert encoded.shape == (500, 4)
    assert np.allclose(encoded[:, 0], signal)
    assert np.allclose(encoded[:, 3], signal)
    assert np.allclose(encoded[:, 1], 0.0, atol=1e-8)
    assert np.allclose(encoded[:, 2], 0.0, atol=1e-8)
    decoded = decode_foa(encoded, "acn_sn3d")
    assert np.allclose(decoded[:, 0], signal)
    assert np.allclose(decoded[:, 1], signal)  # forward / X
    assert np.allclose(decoded[:, 2], 0.0, atol=1e-8)


def test_fuma_w_is_scaled_back_to_sn3d():
    signal = np.sin(np.linspace(0, 8, 400))
    encoded = encode_plane_wave(0.0, 0.0, signal, "fuma")
    assert np.allclose(encoded[:, 0], signal / np.sqrt(2.0))
    assert np.allclose(encoded[:, 1], signal)
    decoded = decode_foa(encoded, "fuma")
    assert np.allclose(decoded[:, 0], signal, atol=1e-8)
    assert np.allclose(decoded[:, 1], signal, atol=1e-8)


def test_acn_n3d_first_order_is_unscaled():
    signal = np.sin(np.linspace(0, 6, 300))
    encoded = encode_plane_wave(90.0, 0.0, signal, "acn_n3d")
    assert np.allclose(encoded[:, 1], np.sqrt(3.0) * signal)
    decoded = decode_foa(encoded, "acn_n3d")
    assert np.allclose(decoded[:, 2], signal, atol=1e-8)  # left / Y
    assert np.allclose(decoded[:, 1], 0.0, atol=1e-7)


def test_elevation_encoding_points_up():
    signal = np.sin(np.linspace(0, 4, 200))
    encoded = encode_plane_wave(40.0, 90.0, signal, "acn_sn3d")
    decoded = decode_foa(encoded, "acn_sn3d")
    assert np.allclose(decoded[:, 3], signal, atol=1e-6)
    assert np.allclose(decoded[:, 1], 0.0, atol=1e-6)
    assert np.allclose(decoded[:, 2], 0.0, atol=1e-6)


def test_decoder_refuses_other_channel_counts():
    with pytest.raises(AnalysisError):
        decode_foa(np.zeros((10, 2)), "acn_sn3d")
    with pytest.raises(AnalysisError):
        decode_foa(np.zeros((10, 4)), "guess")
