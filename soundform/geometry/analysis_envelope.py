"""Smooth display envelopes around pavilion origins, in metres."""

import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator


def merged_envelope(centers, radius, *, merge=True, strength=1.0, full_sphere=False):
    centers = np.asarray(centers, dtype=float)
    radii = np.broadcast_to(np.asarray(radius, dtype=float), (len(centers),))
    if not len(centers) or not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError("Provide finite origin points and radii.")
    if np.any(radii <= 0) or not 0.1 <= strength <= 3:
        raise ValueError("Use positive radii and connection strength between 0.1 and 3.")
    if not merge:
        spheres = [pv.Sphere(radius=float(r), center=center, theta_resolution=56,
                             phi_resolution=56, end_phi=180 if full_sphere else 90) for center, r in zip(centers, radii)]
        return spheres[0].merge(spheres[1:], merge_points=False) if len(spheres) > 1 else spheres[0]
    # Normalize the threshold so strength changes bridges without changing an isolated radius.
    support = 1 + strength
    lower = (centers - radii[:, None] * support * 1.05).min(axis=0)
    upper = (centers + radii[:, None] * support * 1.05).max(axis=0)
    span = upper - lower
    spacing = max(float(radii.min()) / 14, float(span.max()) / 110)
    dimensions = np.maximum(3, np.ceil(span / spacing).astype(int) + 1)
    grid = pv.ImageData(dimensions=dimensions, spacing=(spacing,) * 3, origin=lower)
    points = grid.points
    values = np.zeros(len(points))
    for center, r in zip(centers, radii):
        distance_squared = np.sum((points - center) ** 2, axis=1)
        values += np.maximum(0, 1 - distance_squared / (support * r) ** 2) ** 2
    grid.point_data["influence"] = values
    surface = grid.contour([(1 - 1 / support ** 2) ** 2], scalars="influence")
    if full_sphere:
        return surface
    return surface.clip(normal=(0, 0, 1), origin=(0, 0, centers[:, 2].min()), invert=False)


def envelope_energy(points, centers, azimuth, elevation, scores, rotation):
    """Blend directional samples continuously across the merged surface."""
    centers = np.unique(np.asarray(centers, dtype=float), axis=0)
    angles = np.asarray(azimuth)
    elevation = np.asarray(elevation)
    values = np.clip(np.asarray(scores), 0, 1)
    # Fold downward energy onto the horizon so no lower arrivals are lost.
    if np.any(elevation <= 0):
        horizon = values[elevation <= 0].max(axis=0)
        values = np.vstack((horizon, values[elevation > 0]))
        elevation = np.r_[0, elevation[elevation > 0]]
    extended_angles = np.r_[angles[-1] - 360, angles, angles[0] + 360]
    values = np.column_stack((values[:, -1], values, values[:, 0]))
    interpolate = RegularGridInterpolator((elevation, extended_angles), values,
                                         bounds_error=False, fill_value=None)
    energy, weights = np.zeros(len(points)), np.zeros(len(points))
    for center in centers:
        relative = points - center
        distance = np.linalg.norm(relative, axis=1)
        bearing = (np.degrees(np.arctan2(-relative[:, 0], relative[:, 1])) - rotation) % 360
        altitude = np.degrees(np.arctan2(relative[:, 2], np.linalg.norm(relative[:, :2], axis=1)))
        samples = interpolate(np.column_stack((
            np.clip(altitude, elevation[0], elevation[-1]), bearing
        )))
        weight = 1 / np.maximum(distance, 0.1) ** 4
        energy += samples * weight
        weights += weight
    return np.clip(energy / np.maximum(weights, 1e-30), 0, 1)


def analyze_envelope(surface, threshold_db=-12):
    """Describe actual sampled triangles for inspection and AI design context."""
    mesh = surface.triangulate()
    if not mesh.n_cells or "energy" not in mesh.point_data:
        return None
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    triangles = mesh.points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    energy = np.asarray(mesh.point_data["energy"])
    face_energy = energy[faces].mean(axis=1)
    total_area = max(float(areas.sum()), 1e-30)
    indices = np.unique(np.r_[np.linspace(0, len(energy) - 1, min(48, len(energy))).astype(int),
                              int(energy.argmax())])
    return {
        "interpretation": "Directional recording sampled on the analysis envelope. "
                          "The same recording is used at every origin.",
        "area_m2": round(float(areas.sum()), 3),
        "mean_relative_energy": round(float(areas @ face_energy / total_area), 5),
        "threshold_db_relative_to_recording_peak": threshold_db,
        "above_threshold_area_percent": round(
            float(areas[face_energy >= 10 ** (threshold_db / 10)].sum() / total_area * 100), 2),
        "samples": [{"xyz_m": mesh.points[i].round(3).tolist(),
                     "relative_energy": round(float(energy[i]), 5)} for i in indices],
    }


def clip_to_terrain(surface, terrain=None):
    """Keep the envelope above the terrain and expose its actual intersection rim."""
    if not surface.n_points:
        return surface
    if terrain is None or not terrain.n_cells:
        return surface.clip(normal=(0, 0, 1), origin=(0, 0, 0), invert=False)
    ground = terrain.triangulate().compute_normals(
        consistent_normals=True, auto_orient_normals=False,
        point_normals=False, cell_normals=True, inplace=False)
    if ground.cell_normals[:, 2].mean() < 0:
        ground = ground.flip_faces()
    sampled = surface.compute_implicit_distance(ground, inplace=False)
    return sampled.clip_scalar(scalars="implicit_distance", value=0, invert=False)


def scatter_surface_points(surface, quiet_density, loud_density, seed=42, limit=50000, energy_threshold=0.0):
    """Area-weighted surface samples, with higher density at higher relative energy."""
    if quiet_density < 0 or loud_density < quiet_density:
        raise ValueError("Loud-area density must be at least the nonnegative quiet-area density.")
    if not 0 <= energy_threshold <= 1:
        raise ValueError("Relative-energy threshold must be between 0 and 1.")
    mesh = surface.triangulate()
    if not mesh.n_cells or "energy" not in mesh.point_data:
        raise ValueError("Analyze audio and create a metaball before generating points.")
    if energy_threshold > 0:
        mesh = mesh.clip_scalar(scalars="energy", value=energy_threshold, invert=False).triangulate()
    if not mesh.n_cells:
        return pv.PolyData(), False
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    triangles = mesh.points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    energy = np.clip(np.asarray(mesh.point_data["energy"]), 0, 1)
    density = quiet_density + (loud_density - quiet_density) * energy[faces].mean(axis=1)
    weights = areas * density
    total = float(weights.sum())
    count = min(limit, round(total))
    if count <= 0:
        return pv.PolyData(), False
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=weights / total)
    root = np.sqrt(rng.random(count))
    fraction = rng.random(count)
    barycentric = np.column_stack((1 - root, root * (1 - fraction), root * fraction))
    points = np.einsum("ij,ijk->ik", barycentric, triangles[chosen])
    result = pv.PolyData(points)
    result.point_data["energy"] = (energy[faces[chosen]] * barycentric).sum(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    result.point_data["Normals"] = normals[chosen]
    return result, round(total) > limit
