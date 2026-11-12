"""Heuristic before/after exposure.

For each spherical cell, a ray travels from outside the site toward the
listening point at 1.5 m. If the mesh blocks that ray more than 0.35 m before
the listener, and the hit is above the ground, the cell counts as covered.

Exposure after is incoming relative energy times (1 − coverage). That is an
occlusion fraction, not a decibel reduction, reverberation time, or certified
transmission loss. Reflections, diffraction, absorption, source distance, and
materials are not estimated from the recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from soundform.acoustics.response import ResponseField
from soundform.geometry.design_space import EAR_HEIGHT_M, characteristic_radius, sound_az_el_to_design_vector
from soundform.models.project import DesignParameters

DISCLAIMER = (
    "Heuristic estimate only: directional line-of-sight occlusion at a listening "
    f"height of {EAR_HEIGHT_M:.1f} m. This is not a sound pressure reduction, "
    "not reverberation time, and not a compliance or transmission-loss result. "
    "Real behaviour also depends on reflection, diffraction, absorption, source "
    "distance, and materials, which are not in the recording."
)


@dataclass
class ProtectionReport:
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    energy: np.ndarray
    coverage: np.ndarray
    exposure_before: np.ndarray
    exposure_after: np.ndarray
    band_exposure: dict[str, dict[str, float]]
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


def _band_totals(response: ResponseField, coverage: np.ndarray) -> dict[str, dict[str, float]]:
    """Use the spherical score's energy as the broadband total.

    Per-band comparison uses the same coverage, because one mesh is built.
    The maps passed in `response` are the combined energy; band curves are
    filled by `estimate_protection` from the sound-field aggregates when present.
    """
    before = float(np.sum(response.energy))
    after = float(np.sum(response.energy * (1.0 - coverage)))
    return {"combined": {"before": before, "after": after}}


def estimate_protection(
    mesh: trimesh.Trimesh,
    response: ResponseField,
    params: DesignParameters,
    band_maps: dict[str, np.ndarray] | None = None,
) -> ProtectionReport:
    warnings = [DISCLAIMER]
    azimuth = np.asarray(response.azimuth_deg, dtype=np.float64)
    elevation = np.asarray(response.elevation_deg, dtype=np.float64)
    energy = np.asarray(response.energy, dtype=np.float64)
    if energy.ndim != 2:
        raise ValueError("Protection estimates need a spherical energy map.")

    az_grid, el_grid = np.meshgrid(azimuth, elevation)
    direction_x, direction_y, direction_z = sound_az_el_to_design_vector(
        az_grid, el_grid, params.rotation_offset_deg
    )
    direction = np.stack([direction_x, direction_y, direction_z], axis=-1)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / np.maximum(norm, 1e-8)

    listener = np.array(
        [params.protected_center_x, params.protected_center_y, EAR_HEIGHT_M],
        dtype=np.float64,
    )
    reach = max(characteristic_radius(params), float(params.pavilion_diameter) * 0.5, 8.0) * 3.0
    origins = listener.reshape(1, 1, 3) + direction * reach
    travel = -direction
    n_elev, n_az = energy.shape
    origin_flat = origins.reshape(-1, 3)
    travel_flat = travel.reshape(-1, 3)

    coverage = np.zeros(energy.shape, dtype=np.float64)
    rays_ok = True
    try:
        # All hits, not only the nearest. A ray that clips the ground disc still
        # needs to report a wall or ridge farther along the same line.
        locations, ray_index, _triangles = mesh.ray.intersects_location(
            ray_origins=origin_flat,
            ray_directions=travel_flat,
            multiple_hits=True,
        )
    except Exception as exc:
        rays_ok = False
        warnings.append(
            f"Occlusion rays could not be tested ({exc}). Coverage is unavailable, "
            "so no direction is described as protected or exposed."
        )
        locations = np.zeros((0, 3))
        ray_index = np.zeros((0,), dtype=int)

    if rays_ok and len(locations):
        hit_distance = np.linalg.norm(locations - origin_flat[ray_index], axis=1)
        target_distance = np.linalg.norm(listener - origin_flat[ray_index], axis=1)
        # Ignore the base surface. A hit counts only when it stands above the
        # ground and still lies in front of the listening point.
        valid = (hit_distance < target_distance - 0.35) & (locations[:, 2] > 0.3)
        covered = np.unique(ray_index[valid])
        flat = coverage.reshape(-1)
        flat[covered] = 1.0
        coverage = flat.reshape(energy.shape)

    exposure_before = energy
    exposure_after = energy * (1.0 - coverage)

    horizontal = np.clip(np.cos(np.deg2rad(elevation)), 0.0, None)
    azimuth_energy = (energy * horizontal[:, None]).sum(axis=0)
    azimuth_cover = (coverage * horizontal[:, None]).sum(axis=0) / (horizontal.sum() + 1e-12)
    peak = float(np.max(azimuth_energy)) if azimuth_energy.size else 0.0
    if rays_ok and peak > 1e-8:
        exposed = (azimuth_energy > 0.65 * peak) & (azimuth_cover < 0.25)
        exposed_az = [f"{azimuth[index]:.0f}°" for index in np.flatnonzero(exposed)]
        if exposed_az:
            listed = ", ".join(exposed_az[:8])
            extra = "" if len(exposed_az) <= 8 else f" and {len(exposed_az) - 8} more"
            warnings.append(
                "Loud incoming energy is largely unobstructed near azimuth "
                f"{listed}{extra}. This is a design warning from the occlusion heuristic, "
                "not a measured level."
            )
    if float(np.max(energy)) <= 1e-8:
        warnings.append("The analysed interval has no directional energy to compare.")

    band_exposure: dict[str, dict[str, float]] = {}
    if band_maps:
        for name, raw in band_maps.items():
            band = np.asarray(raw, dtype=np.float64)
            if band.shape != coverage.shape:
                continue
            peak_band = float(np.max(band))
            scaled = np.zeros_like(band) if peak_band <= 1e-20 else band / peak_band
            band_exposure[name] = {
                "before": float(np.mean(scaled)),
                "after": float(np.mean(scaled * (1.0 - coverage))),
            }
    else:
        band_exposure = _band_totals(response, coverage)

    return ProtectionReport(
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        energy=energy,
        coverage=coverage,
        exposure_before=exposure_before,
        exposure_after=exposure_after,
        band_exposure=band_exposure,
        warnings=warnings,
    )
