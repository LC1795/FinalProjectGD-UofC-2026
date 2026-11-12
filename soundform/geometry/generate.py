"""Dispatch the three architectural modes.

Audio analysis does not call this module. The mesh is a design response to an
already computed directional score.
"""

from __future__ import annotations

import trimesh

from soundform.acoustics.response import ResponseField, compute_response
from soundform.acoustics.sound_field import SoundField
from soundform.errors import GeometryError
from soundform.geometry.design_space import prepare_parameters
from soundform.geometry.pavilion import build_pavilion
from soundform.geometry.terrain import build_terrain
from soundform.geometry.walls import build_walls
from soundform.models.project import DesignParameters


def generate_architecture(
    sound_field: SoundField,
    params: DesignParameters,
) -> tuple[trimesh.Trimesh, ResponseField, DesignParameters, list[str]]:
    used, warnings = prepare_parameters(params)
    weights = {
        "broadband": used.weight_broadband,
        "low": used.weight_low,
        "mid": used.weight_mid,
        "high": used.weight_high,
        "persistence": used.weight_persistence,
        "transient": used.weight_transient,
        "confidence": used.weight_confidence,
    }
    response = compute_response(sound_field, weights, smoothing=used.smoothing if used.mode != "terrain" else 0.0)
    warnings.extend(response.warnings)
    try:
        if used.mode == "terrain":
            mesh = build_terrain(response, used)
            warnings.append(
                "Terrain rises toward directional energy. The protected zone is held at the minimum height."
            )
        elif used.mode == "walls":
            mesh = build_walls(response, used)
            warnings.append(
                "Wall heights follow the directional score. Coverage is a line-of-sight heuristic, "
                "not a transmission-loss result."
            )
        elif used.mode == "pavilion":
            mesh = build_pavilion(response, used, warnings)
            warnings.append(
                "The pavilion thickens enclosure toward louder directions and keeps an opening toward the quietest one."
            )
        else:
            raise GeometryError("Choose terrain, walls, or pavilion.")
    except GeometryError:
        raise
    except ValueError as exc:
        raise GeometryError(str(exc)) from exc
    return mesh, response, used, warnings
