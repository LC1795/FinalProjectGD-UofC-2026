"""Import a layered Rhino site into a metre-based frame at the pavilion origin."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_MESH_LAYERS = ("Terrain", "Solids", "Vegetation", "Circulation")
_LAYERS = {name.casefold(): name for name in (*_MESH_LAYERS, "Sound Sources", "Pavilion Origin")}
_METRES = {
    "Microns": 1e-6, "Millimeters": 0.001, "Centimeters": 0.01, "Decimeters": 0.1,
    "Meters": 1.0, "Dekameters": 10.0, "Hectometers": 100.0, "Kilometers": 1000.0,
    "Inches": 0.0254, "Feet": 0.3048, "Yards": 0.9144, "Miles": 1609.344,
    "Microinches": 2.54e-8, "Mils": 2.54e-5, "PrinterPoints": 0.0254 / 72,
    "PrinterPicas": 0.0254 / 6,
}


@dataclass
class SiteModel:
    path: Path
    origin_m: np.ndarray
    meshes: dict[str, tuple[np.ndarray, np.ndarray]]
    sources: list[tuple[str, np.ndarray]]
    warnings: list[str]
    origins: list[tuple[str, np.ndarray]] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)


def _xyz(point):
    return point.X, point.Y, point.Z


def _curve_vertices(curve, scale):
    """Preserve polylines exactly; approximate other Rhino curves for display."""
    polyline = curve.TryGetPolyline()
    if polyline is not None:
        coords = np.asarray([_xyz(p) for p in polyline], dtype=float)
    else:
        domain = curve.Domain
        # Seed every curve span generously, then refine bends to 1 mm chord error.
        count = min(4096, max(32, curve.SpanCount * 8))
        ts = np.linspace(domain.T0, domain.T1, count + 1)
        coords = []
        def sample(a, b, pa, pb, depth=0):
            params = a + (b-a)*np.array([.25, .5, .75])
            q = np.array([_xyz(curve.PointAt(float(t))) for t in params])
            chord = pb-pa
            den = float(chord@chord)
            fraction = np.clip((q-pa)@chord/max(den, 1e-30), 0, 1)
            error = np.linalg.norm(q-(pa+fraction[:, None]*chord), axis=1).max()*scale
            if error > .001 and depth < 8:
                mid = (a+b)/2
                sample(a, mid, pa, q[1], depth+1)
                sample(mid, b, q[1], pb, depth+1)
            else:
                coords.append(pa)
        for a, b in zip(ts[:-1], ts[1:]):
            sample(a, b, np.array(_xyz(curve.PointAt(float(a)))),
                   np.array(_xyz(curve.PointAt(float(b)))))
        coords.append(np.array(_xyz(curve.PointAtEnd)))
        coords = np.asarray(coords, dtype=float)
    if len(coords) < 2 or not np.isfinite(coords).all():
        raise ValueError("Circulation contains an invalid or empty Rhino curve.")
    if curve.IsClosed and not np.allclose(coords[0], coords[-1], rtol=0, atol=1e-12):
        coords = np.vstack([coords, coords[0]])
    return coords


def load_site(path: str | Path) -> SiteModel:
    try:
        import rhino3dm as rhino
    except ImportError as exc:
        raise ValueError("Rhino site import requires rhino3dm. Install it into Soundform's Python environment.") from exc

    path = Path(path).resolve()
    model = rhino.File3dm.Read(str(path))
    if model is None:
        raise ValueError("Could not read this Rhino .3dm file.")
    unit = model.Settings.ModelUnitSystem.name
    scale = _METRES.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported Rhino model units ({unit}). Set explicit model units in Rhino and save again.")
    layers = {layer.Index: layer.FullPath for layer in model.Layers}
    origins = []
    objects = []
    sources = []
    warnings = []
    vertices = {name: [] for name in _MESH_LAYERS}
    faces = {name: [] for name in _MESH_LAYERS}
    offsets = dict.fromkeys(_MESH_LAYERS, 0)
    skipped = {}
    for obj in model.Objects:
        layer_path = layers.get(obj.Attributes.LayerIndex, "")
        category = next((_LAYERS[x.casefold()] for x in layer_path.split("::")
                         if x.casefold() in _LAYERS), None)
        if category is None:
            continue
        geom = obj.Geometry
        if category in vertices and isinstance(geom, rhino.Mesh) and geom.Faces.Count:
            coords = np.array([_xyz(v) for v in geom.Vertices], dtype=float)
            if not np.isfinite(coords).all():
                raise ValueError(f"The {category} layer contains a mesh with invalid coordinates.")
            # .3dm mesh faces encode triangles with a repeated final index.
            connectivity = np.array([tuple(face) for face in geom.Faces], dtype=np.int64)
            if connectivity.min() < 0 or connectivity.max() >= len(coords):
                raise ValueError(f"The {category} layer contains an invalid mesh face.")
            quads = connectivity[:, 2] != connectivity[:, 3]
            triangles = np.concatenate((connectivity[:, :3], connectivity[quads][:, (0, 2, 3)]))
            objects.append({"name": obj.Attributes.Name or f"{category} {len(objects) + 1}",
                            "layer": layer_path, "category": category,
                            "vertices": coords, "faces": triangles})
            faces[category].append(triangles + offsets[category])
            vertices[category].append(coords)
            offsets[category] += len(coords)
        elif category == "Circulation" and isinstance(geom, rhino.Curve):
            coords = _curve_vertices(geom, scale)
            objects.append({"name": obj.Attributes.Name or f"Circulation {len(objects) + 1}",
                            "layer": layer_path, "category": category, "vertices": coords,
                            "faces": np.empty((0, 3), dtype=np.int64),
                            "lines": np.r_[len(coords), np.arange(len(coords))].astype(np.int64)})
        elif category == "Pavilion Origin" and isinstance(geom, rhino.Point):
            origins.append((obj.Attributes.Name or f"Pavilion origin {len(origins) + 1}", _xyz(geom.Location)))
        elif category == "Sound Sources" and isinstance(geom, rhino.Point):
            sources.append((obj.Attributes.Name or "Sound source", _xyz(geom.Location)))
        else:
            skipped[category] = skipped.get(category, 0) + 1
    if not origins:
        raise ValueError("Place at least one Rhino point on the Pavilion Origin layer.")
    origin = np.asarray(origins[0][1], dtype=float)
    if any(not np.isfinite(point).all() for _, point in origins) or any(not np.isfinite(point).all() for _, point in sources):
        raise ValueError("A pavilion or sound-source point has invalid coordinates.")
    for category, count in skipped.items():
        expected = ("Rhino curves or mesh objects" if category == "Circulation" else
                    "mesh objects" if category in _MESH_LAYERS else "Rhino points")
        warnings.append(f"{category}: skipped {count} unsupported object(s). Use {expected}.")
    meshes = {name: ((np.concatenate(vertices[name]) - origin) * scale, np.concatenate(faces[name]))
              for name in _MESH_LAYERS if vertices[name]}
    if "Terrain" not in meshes:
        warnings.append("No terrain mesh was found. The default ground grid is visible.")
    for item in objects:
        item["vertices"] = (item["vertices"] - origin) * scale
    return SiteModel(path, origin * scale, meshes,
                     [(name, (np.asarray(point) - origin) * scale) for name, point in sources], warnings,
                     [(name, (np.asarray(point) - origin) * scale) for name, point in origins], objects)
