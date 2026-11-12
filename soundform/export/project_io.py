"""Save and reload a Soundform project.

The JSON file stores paths and parameters. It does not store audio samples.
A companion .analysis.npz holds the derived directional field when one exists,
and a companion .obj holds the last mesh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from soundform.acoustics.sound_field import SoundField
from soundform.errors import ExportError
from soundform.export.mesh_export import export_mesh
from soundform.models.project import AnalysisSettings, DesignParameters


@dataclass
class ProjectRecord:
    audio_path: str | None = None
    audio_found: bool = False
    convention: str | None = None
    analysis_settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    design: DesignParameters = field(default_factory=DesignParameters)
    sound_field: SoundField | None = None
    mesh: trimesh.Trimesh | None = None
    warnings: list[str] = field(default_factory=list)


def _resolve_audio(stored: str | None, project_path: Path) -> tuple[str | None, bool]:
    if not stored:
        return None, False
    direct = Path(stored)
    if direct.is_file():
        return str(direct), True
    beside = project_path.parent / direct.name
    if beside.is_file():
        return str(beside), True
    return str(direct), False


def save_project(
    path: str,
    audio_path: str | None,
    convention: str | None,
    analysis_settings: AnalysisSettings,
    design: DesignParameters,
    sound_field: SoundField | None = None,
    mesh: trimesh.Trimesh | None = None,
) -> None:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    analysis_name = None
    mesh_name = None
    if sound_field is not None:
        analysis_file = destination.with_suffix(".analysis.npz")
        sound_field.to_npz(str(analysis_file))
        analysis_name = analysis_file.name
    if mesh is not None:
        mesh_file = destination.with_suffix(".mesh.obj")
        export_mesh(mesh, str(mesh_file))
        mesh_name = mesh_file.name
    payload = {
        "format": "soundform-project",
        "version": 1,
        "audio_path": audio_path,
        "convention": convention,
        "analysis_settings": analysis_settings.to_dict(),
        "design": design.to_dict(),
        "analysis_cache": analysis_name,
        "mesh_cache": mesh_name,
        "notes": [
            "Audio samples are not embedded. audio_path is a reference only.",
            "Levels in the analysis cache are relative, not dB SPL.",
            "Protection results are not stored as certified acoustic metrics.",
        ],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project(path: str) -> ProjectRecord:
    source = Path(path)
    if not source.is_file():
        raise ExportError("That project file does not exist.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"The project file is not valid JSON. ({exc})") from exc
    if payload.get("format") != "soundform-project":
        raise ExportError("This JSON file is not a Soundform project.")
    record = ProjectRecord(
        convention=payload.get("convention"),
        analysis_settings=AnalysisSettings.from_dict(payload.get("analysis_settings") or {}),
        design=DesignParameters.from_dict(payload.get("design") or {}),
    )
    resolved, found = _resolve_audio(payload.get("audio_path"), source)
    record.audio_path = resolved
    record.audio_found = found
    if payload.get("audio_path") and not found:
        record.warnings.append(
            f"The source recording was not found at {payload.get('audio_path')}. "
            "Parameters are restored. Import the WAV again to re-analyse it."
        )
    cache_name = payload.get("analysis_cache")
    if cache_name:
        cache_path = source.parent / cache_name
        if cache_path.is_file():
            try:
                record.sound_field = SoundField.from_npz(str(cache_path))
            except Exception as exc:
                record.warnings.append(f"Saved analysis could not be read ({exc}).")
        else:
            record.warnings.append("The saved analysis cache is missing.")
    mesh_name = payload.get("mesh_cache")
    if mesh_name:
        mesh_path = source.parent / mesh_name
        if mesh_path.is_file():
            try:
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
                if len(loaded.faces) == 0:
                    raise ExportError("The saved mesh is empty.")
                record.mesh = loaded
            except Exception as exc:
                record.warnings.append(f"Saved geometry could not be read ({exc}).")
        else:
            record.warnings.append("The saved mesh file is missing.")
    return record


def export_analysis_json(path: str, sound_field: SoundField, report_rows: list[dict] | None = None) -> None:
    destination = Path(path)
    payload = {
        "disclaimer": sound_field.level_note,
        "convention": sound_field.convention,
        "interval_s": list(sound_field.interval),
        "rms_dbfs": sound_field.rms_dbfs,
        "peak_dbfs": sound_field.peak_dbfs,
        "band_limits_hz": {key: list(value) for key, value in sound_field.band_limits.items()},
        "warnings": sound_field.warnings,
        "times_s": sound_field.times.tolist(),
        "dominant_azimuth_deg": [None if not np.isfinite(value) else float(value) for value in sound_field.dominant_az],
        "dominant_elevation_deg": [None if not np.isfinite(value) else float(value) for value in sound_field.dominant_el],
        "directional_confidence": sound_field.bin_confidence.astype(float).tolist(),
        "directions": report_rows or [],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_analysis_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ExportError("There is no analysis table to export yet.")
    destination = Path(path)
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analysis_rows(sound_field: SoundField, coverage: np.ndarray | None = None) -> list[dict]:
    """One row per spherical cell of the time-averaged relative field."""
    energy = sound_field.aggregate("broadband")
    low = sound_field.aggregate("low")
    mid = sound_field.aggregate("mid")
    high = sound_field.aggregate("high")
    peak = float(np.max(energy)) if energy.size else 0.0
    scale = peak if peak > 1e-20 else 1.0
    rows: list[dict] = []
    for el_index, elevation in enumerate(sound_field.elevation_deg):
        for az_index, azimuth in enumerate(sound_field.azimuth_deg):
            relative = float(energy[el_index, az_index] / scale)
            covered = 0.0 if coverage is None else float(coverage[el_index, az_index])
            rows.append(
                {
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": float(elevation),
                    "relative_broadband": relative,
                    "relative_low": float(low[el_index, az_index] / (np.max(low) + 1e-20)),
                    "relative_mid": float(mid[el_index, az_index] / (np.max(mid) + 1e-20)),
                    "relative_high": float(high[el_index, az_index] / (np.max(high) + 1e-20)),
                    "coverage": covered,
                    "exposure_before": relative,
                    "exposure_after": relative * (1.0 - covered),
                }
            )
    return rows
