import json
from pathlib import Path

import numpy as np
import trimesh

from soundform.errors import ExportError
from soundform.export.mesh_export import export_mesh
from soundform.export.project_io import analysis_rows, export_analysis_csv, load_project, save_project
from soundform.geometry.generate import generate_architecture
from soundform.models.project import AnalysisSettings, DesignParameters
from soundform.tests.helpers import blob_field
import pytest


def test_obj_and_stl_roundtrip(tmp_path: Path):
    mesh, _response, _used, _warnings = generate_architecture(blob_field(0.0), DesignParameters(mode="terrain", grid_resolution=28, smoothing=0.0))
    obj = export_mesh(mesh, str(tmp_path / "form.obj"))
    stl = export_mesh(mesh, str(tmp_path / "form.stl"))
    assert obj["faces"] > 10
    assert stl["vertices"] > 10
    assert Path(obj["path"]).stat().st_size > 100
    reloaded = trimesh.load(obj["path"], force="mesh", process=False)
    assert len(reloaded.faces) > 10


def test_invalid_meshes_are_rejected(tmp_path: Path):
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    with pytest.raises(ExportError):
        export_mesh(empty, str(tmp_path / "empty.obj"))
    broken = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, np.nan]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    with pytest.raises(ExportError):
        export_mesh(broken, str(tmp_path / "nan.stl"))
    with pytest.raises(ExportError):
        export_mesh(broken, str(tmp_path / "form.ply"))


def test_project_stores_a_reference_not_the_audio(tmp_path: Path):
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 80 + b"NOT-EMBEDDED-AUDIO-PAYLOAD")
    field = blob_field(25.0, 5.0)
    mesh, _response, _used, _warnings = generate_architecture(field, DesignParameters(mode="terrain", grid_resolution=24, smoothing=0.1))
    project = tmp_path / "study.json"
    save_project(
        str(project),
        str(audio),
        "acn_sn3d",
        AnalysisSettings(start_s=0.2, end_s=1.4),
        DesignParameters(mode="walls", max_height=6.5, rotation_offset_deg=15),
        field,
        mesh,
    )
    text = project.read_text(encoding="utf-8")
    assert "NOT-EMBEDDED-AUDIO-PAYLOAD" not in text
    assert len(text) < 80_000
    payload = json.loads(text)
    assert payload["audio_path"].endswith("recording.wav")
    assert payload["analysis_cache"].endswith(".npz")
    loaded = load_project(str(project))
    assert loaded.audio_found
    assert loaded.convention == "acn_sn3d"
    assert loaded.design.mode == "walls"
    assert loaded.design.max_height == pytest.approx(6.5)
    assert loaded.analysis_settings.start_s == pytest.approx(0.2)
    assert loaded.sound_field is not None
    assert np.allclose(loaded.sound_field.dominant_az, field.dominant_az)
    assert loaded.mesh is not None
    assert len(loaded.mesh.faces) > 0


def test_missing_audio_does_not_drop_the_project(tmp_path: Path):
    project = tmp_path / "gone.json"
    save_project(
        str(project),
        str(tmp_path / "missing.wav"),
        "fuma",
        AnalysisSettings(),
        DesignParameters(protected_radius=7.0),
        None,
        None,
    )
    loaded = load_project(str(project))
    assert not loaded.audio_found
    assert loaded.design.protected_radius == pytest.approx(7.0)
    assert loaded.convention == "fuma"
    assert loaded.warnings


def test_analysis_csv_and_json_rows(tmp_path: Path):
    field = blob_field()
    rows = analysis_rows(field)
    assert rows
    assert {"azimuth_deg", "relative_broadband", "exposure_after"} <= set(rows[0])
    csv_path = tmp_path / "analysis.csv"
    export_analysis_csv(str(csv_path), rows[:5])
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "azimuth_deg" in header
    assert "dB SPL" not in csv_path.read_text(encoding="utf-8")
