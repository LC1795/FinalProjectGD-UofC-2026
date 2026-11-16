"""Import a layered Rhino site into a metre-based frame at the pavilion origin."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MESH_LAYERS = ("Terrain", "Solids", "Vegetation")
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


def _xyz(point):
    return point.X, point.Y, point.Z


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
            faces[category].append(triangles + offsets[category])
            vertices[category].append(coords)
            offsets[category] += len(coords)
        elif category == "Pavilion Origin" and isinstance(geom, rhino.Point):
            origins.append(_xyz(geom.Location))
        elif category == "Sound Sources" and isinstance(geom, rhino.Point):
            sources.append((obj.Attributes.Name or "Sound source", _xyz(geom.Location)))
        else:
            skipped[category] = skipped.get(category, 0) + 1
    if len(origins) != 1:
        raise ValueError(f"Place exactly one Rhino point on the Pavilion Origin layer. Found {len(origins)}.")
    origin = np.asarray(origins[0], dtype=float)
    if not np.isfinite(origin).all() or any(not np.isfinite(point).all() for _, point in sources):
        raise ValueError("A pavilion or sound-source point has invalid coordinates.")
    for category, count in skipped.items():
        expected = "mesh objects" if category in _MESH_LAYERS else "Rhino points"
        warnings.append(f"{category}: skipped {count} unsupported object(s). Use {expected}.")
    meshes = {name: ((np.concatenate(vertices[name]) - origin) * scale, np.concatenate(faces[name]))
              for name in _MESH_LAYERS if vertices[name]}
    if "Terrain" not in meshes:
        warnings.append("No terrain mesh was found. The default ground grid is visible.")
    return SiteModel(path, origin * scale, meshes,
                     [(name, (np.asarray(point) - origin) * scale) for name, point in sources], warnings)
