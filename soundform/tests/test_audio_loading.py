import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from soundform.audio.format_detection import append_axml_chunk
from soundform.audio.loader import load_audio
from soundform.audio.playback import stereo_monitor_downmix
from soundform.audio.synthetic import encode_plane_wave, write_wav
from soundform.errors import AudioLoadError


def _write_extensible(path: Path, samples: np.ndarray, sample_rate: int, channel_mask: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    frames, channels = pcm.shape
    block_align = channels * 2
    fmt = struct.pack("<HHIIHH", 0xFFFE, channels, sample_rate, sample_rate * block_align, block_align, 16)
    fmt += struct.pack("<HHI", 22, 16, channel_mask)
    guid = struct.pack("<IHH", 0x00000001, 0x0000, 0x0010) + bytes.fromhex("800000aa00389b71")
    fmt += guid

    def chunk(chunk_id: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) % 2 else b""
        return chunk_id + struct.pack("<I", len(payload)) + payload + pad

    body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"data", pcm.tobytes())
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def test_plain_four_channel_file_is_not_assumed_ambisonic(tmp_path: Path):
    samples = np.random.default_rng(0).normal(0.0, 0.1, (1600, 4)).astype(np.float32)
    path = tmp_path / "quad.wav"
    write_wav(path, samples, 16000)
    loaded = load_audio(path)
    assert loaded.channels == 4
    assert loaded.detection.spatial_status == "needs_convention"
    assert loaded.detection.suggested_convention is None
    assert "not assumed" in loaded.detection.messages[-1].lower() or "not identified" in " ".join(loaded.detection.messages).lower()


def test_ambix_metadata_suggests_acn_sn3d(tmp_path: Path):
    samples = np.zeros((800, 4), dtype=np.float32)
    path = tmp_path / "ambix.wav"
    write_wav(path, samples, 16000)
    append_axml_chunk(path, "<format>ambix</format><ordering>ACN</ordering><normalization>SN3D</normalization>")
    loaded = load_audio(path)
    assert loaded.detection.suggested_convention == "acn_sn3d"
    assert loaded.detection.spatial_status == "ready"


def test_conflicting_metadata_is_not_guessed(tmp_path: Path):
    path = tmp_path / "conflict.wav"
    write_wav(path, np.zeros((400, 4), dtype=np.float32), 16000)
    append_axml_chunk(path, "<format>ambix</format><normalization>FuMa</normalization>")
    loaded = load_audio(path)
    assert loaded.detection.suggested_convention is None
    assert any("more than one" in message for message in loaded.detection.messages)


def test_speaker_mask_is_not_treated_as_ambisonic(tmp_path: Path):
    path = tmp_path / "quad_speakers.wav"
    samples = np.zeros((600, 4), dtype=np.float32)
    _write_extensible(path, samples, 16000, 0x33)
    loaded = load_audio(path)
    assert loaded.detection.suggested_convention is None
    text = " ".join(loaded.detection.messages).lower()
    assert "loudspeaker" in text or "speaker" in text


def test_mono_and_stereo_are_flagged_without_claiming_full_spatial_detail(tmp_path: Path):
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    sf.write(mono, np.zeros(800, dtype=np.float32), 16000)
    sf.write(stereo, np.zeros((800, 2), dtype=np.float32), 16000)
    mono_info = load_audio(mono)
    stereo_info = load_audio(stereo)
    assert mono_info.detection.spatial_status == "unsupported"
    assert "no directional" in " ".join(mono_info.detection.messages).lower()
    stereo_text = " ".join(stereo_info.detection.messages).lower()
    assert stereo_info.detection.spatial_status == "unsupported"
    assert "not treated as ambisonics" in stereo_text
    assert "does not contain" in stereo_text
    assert "binaural" in stereo_text


def test_unsupported_channel_count_is_not_trimmed(tmp_path: Path):
    path = tmp_path / "hoa.wav"
    write_wav(path, np.zeros((300, 9), dtype=np.float32), 16000)
    loaded = load_audio(path)
    assert loaded.channels == 9
    assert loaded.detection.spatial_status == "unsupported"
    assert "not trimmed" in " ".join(loaded.detection.messages).lower()


def test_corrupt_empty_and_invalid_rate(tmp_path: Path):
    garbage = tmp_path / "broken.wav"
    garbage.write_bytes(b"not a wav file at all, just text pretending")
    with pytest.raises(AudioLoadError):
        load_audio(garbage)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioLoadError, match="empty|too small"):
        load_audio(empty)
    missing = tmp_path / "missing.wav"
    with pytest.raises(AudioLoadError):
        load_audio(missing)
    foreign = tmp_path / "clip.mp3"
    foreign.write_bytes(b"ID3" + b"\x00" * 64)
    with pytest.raises(AudioLoadError, match="Unsupported format"):
        load_audio(foreign)


def test_nan_samples_are_rejected(tmp_path: Path):
    path = tmp_path / "nan.wav"
    samples = np.zeros((400, 4), dtype=np.float32)
    samples[5, 0] = np.nan
    sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(AudioLoadError, match="NaN"):
        load_audio(path)


def test_bit_depth_and_metadata_are_reported(tmp_path: Path):
    path = tmp_path / "pcm24.wav"
    samples = encode_plane_wave(0, 0, np.sin(np.linspace(0, 20, 1200)), "acn_sn3d").astype(np.float32)
    write_wav(path, samples * 0.2, 16000, subtype="PCM_24")
    loaded = load_audio(path)
    assert "24" in loaded.bit_depth
    assert loaded.sample_rate == 16000
    assert loaded.duration > 0
    assert loaded.channel_stats[0].peak_dbfs > -120


def test_monitor_downmix_respects_left_and_front():
    tone = np.sin(np.linspace(0, 80, 4000))
    front = stereo_monitor_downmix(
        np.column_stack(
            [
                tone,
                tone,  # X forward
                np.zeros_like(tone),
                np.zeros_like(tone),
            ]
        )
    )
    left_energy = np.sqrt(np.mean(front[:, 0] ** 2))
    right_energy = np.sqrt(np.mean(front[:, 1] ** 2))
    assert left_energy == pytest.approx(right_energy, rel=0.05)

    left = stereo_monitor_downmix(
        np.column_stack(
            [
                tone,
                np.zeros_like(tone),
                tone,  # Y left
                np.zeros_like(tone),
            ]
        )
    )
    assert np.sqrt(np.mean(left[:, 0] ** 2)) > np.sqrt(np.mean(left[:, 1] ** 2)) * 1.5
