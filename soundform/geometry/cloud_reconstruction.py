"""Delaunay and uniform-grid Poisson reconstruction of oriented cloud samples."""

import numpy as np
import pyvista as pv
from scipy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree, Delaunay, QhullError


def reconstruct_cloud(points, method, alpha=0.0, resolution=64, boundary=None, voronoi_cells=250, voronoi_gap=0.08, voronoi_seed=42, max_edge=0.0, projection="xy", voronoi_max_reach=0.0, circulation_curves=None, preserve_circulation=True):
    if points is None or points.n_points < 12:
        raise ValueError("Generate at least 12 cloud surface points first.")
    xyz = np.asarray(points.points)
    if not np.isfinite(xyz).all():
        raise ValueError("Cloud point coordinates must be finite.")
    if method != "delaunay" and np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 3:
        raise ValueError("The points must span a three-dimensional surface.")
    if method == "delaunay":
        if boundary is not None:
            from soundform.geometry.surface_triangulation import triangulate_surface_points
            mesh = triangulate_surface_points(xyz, boundary, max_edge)
        else:
            mesh = _planar_delaunay(xyz, max_edge, projection)
    elif method == "voronoi":
        energy = (np.asarray(points.point_data["energy"]) if "energy" in points.point_data
                  else np.zeros(len(xyz)))
        mesh = _voronoi(xyz, boundary, voronoi_cells, voronoi_gap, voronoi_seed,
                        voronoi_max_reach, energy)
        if preserve_circulation and circulation_curves:
            from soundform.geometry.voronoi_filters import subtract_circulation
            mesh = subtract_circulation(mesh, circulation_curves)
            if not mesh.n_cells:
                raise ValueError("Circulation clearances removed every Voronoi tile. Reduce their radii.")
    elif method == "poisson":
        if "Normals" not in points.point_data:
            raise ValueError("Regenerate the surface points to include their oriented normals.")
        mesh = _poisson(xyz, np.asarray(points.point_data["Normals"]), resolution)
    else:
        raise ValueError("Unknown reconstruction method.")
    if not mesh.n_cells:
        raise ValueError("No triangles remain. Increase the maximum edge distance or generate more points.")
    if method == "delaunay":
        return mesh
    _, nearest = cKDTree(xyz).query(mesh.points)
    source = mesh.point_data["source_point"] if method == "delaunay" else nearest
    mesh.point_data["energy"] = (np.asarray(points.point_data["energy"])[source]
                                  if "energy" in points.point_data else np.zeros(mesh.n_points))
    return mesh


def _planar_delaunay(xyz, max_edge, projection):
    """One global 2D triangulation, lifted to the original 3D coordinates."""
    if not np.isfinite(max_edge) or max_edge < 0:
        raise ValueError("Maximum edge distance must be nonnegative (0 means unlimited).")
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection == "best_fit":
        centered = xyz - xyz.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        uv = centered @ basis[:2].T
    elif projection in axes:
        uv = xyz[:, axes[projection]]
    else:
        raise ValueError("Choose XY, XZ, YZ, or best-fit projection.")
    # Coincident projected samples cannot both belong to a planar triangulation.
    # Retain the first sample deterministically, including its original elevation.
    uv, source = np.unique(uv, axis=0, return_index=True)
    if len(uv) < 3 or np.linalg.matrix_rank(uv - uv.mean(axis=0)) < 2:
        raise ValueError("Points are collinear in this projection. Choose a different plane.")
    try:
        triangles = Delaunay(uv).simplices
    except QhullError as error:
        raise ValueError("Cannot triangulate this projection. Choose a different plane.") from error
    vertices = xyz[source]
    if max_edge > 0:
        corners = vertices[triangles]
        lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
        triangles = triangles[np.all(lengths <= max_edge, axis=1)]
    if not len(triangles):
        raise ValueError("No triangles meet the maximum edge distance. Increase the limit.")
    used, inverse = np.unique(triangles, return_inverse=True)
    mesh = pv.PolyData(vertices[used], np.c_[np.full(len(triangles), 3), inverse.reshape(-1, 3)].ravel())
    mesh.point_data["source_point"] = source[used]
    mesh.field_data["projection"] = np.array([projection])
    mesh.field_data["projected_duplicates_removed"] = np.array([len(xyz) - len(uv)])
    return mesh


def _poisson(points, normals, resolution):
    """Solve Laplacian(chi)=div(V), then contour at the samples' mean chi."""
    resolution = int(resolution)
    if not 32 <= resolution <= 128 or not np.isfinite(normals).all():
        raise ValueError("Use grid resolution 32–128 and finite oriented normals.")
    span = float(np.ptp(points, axis=0).max())
    side = span * 1.5
    spacing = side / (resolution - 1)
    origin = (points.min(axis=0) + points.max(axis=0)) / 2 - side / 2
    coordinates = (points - origin) / spacing
    base = np.floor(coordinates).astype(int)
    fraction = coordinates - base
    vector = np.zeros((3, resolution, resolution, resolution))
    mass = np.zeros((resolution,) * 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz])
                index = base + offset
                weight = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                address = tuple(index.T)
                np.add.at(mass, address, weight)
                for axis in range(3):
                    np.add.at(vector[axis], address, weight * normals[:, axis])
    mass = gaussian_filter(mass, 1)
    for axis in range(3):
        vector[axis] = gaussian_filter(vector[axis], 1) / np.maximum(mass, 0.05)
    divergence = sum((np.roll(vector[axis], -1, axis=axis)
                      - np.roll(vector[axis], 1, axis=axis)) / (2 * spacing)
                     for axis in range(3))
    eigenvalues = -4 * np.sin(np.pi * np.fft.fftfreq(resolution)) ** 2 / spacing ** 2
    laplacian = (eigenvalues[:, None, None] + eigenvalues[None, :, None]
                 + eigenvalues[None, None, :])
    laplacian[0, 0, 0] = 1
    spectrum = fftn(divergence) / laplacian
    spectrum[0, 0, 0] = 0
    field = ifftn(spectrum).real
    isovalue = float(map_coordinates(field, coordinates.T, order=1).mean())
    grid = pv.ImageData(dimensions=(resolution,) * 3, spacing=(spacing,) * 3, origin=origin)
    grid.point_data["poisson"] = field.ravel(order="F")
    return grid.contour([isovalue], scalars="poisson").triangulate()


def _voronoi(points, boundary, cell_count=250, gap=0.08, seed=42, max_reach=0.0, energy=None):
    """Restricted Euclidean Voronoi patches on the actual triangulated surface.

    Each surface triangle is intersected with nearest-site half-spaces. No
    volumetric walls, projection to a plane, or closure across the terrain rim.
    """
    if boundary is None or not boundary.n_cells:
        raise ValueError("Create a cloud before generating its Voronoi mesh.")
    if not np.isfinite(gap) or gap < 0 or not 0 <= cell_count <= 5000:
        raise ValueError("Use a nonnegative gap and 0-5000 cells (0 uses all points).")
    if not np.isfinite(max_reach) or max_reach < 0:
        raise ValueError("Maximum cell reach must be nonnegative (0 means unlimited).")
    sites, source = np.unique(np.asarray(points, dtype=float), axis=0, return_index=True)
    if cell_count and cell_count < len(sites):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(sites), int(cell_count), replace=False))
        sites, source = sites[selected], source[selected]
    tree = cKDTree(sites)
    surface = boundary.triangulate()
    triangles = np.asarray(surface.points, dtype=float)[surface.faces.reshape(-1, 4)[:, 1:]]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    nearest, _ = tree.query(centers)
    # A winner anywhere on a triangle is at most nearest(center)+2*radius
    # from its center. Include extra margin for inset cell borders.
    search_radius = nearest + 2 * radii + 2 * gap + 1e-9
    candidates = tree.query_ball_point(centers, search_radius)
    fragments = {}
    for triangle, nearby in zip(triangles, candidates):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= max(np.linalg.norm(normal), 1e-15)
        for index in nearby:
            polygon = triangle.copy()
            for other in nearby:
                if other == index:
                    continue
                direction = sites[other] - sites[index]
                midpoint = (sites[other] + sites[index]) / 2
                projected = direction - normal * np.dot(direction, normal)
                offset = gap * 0.5 * np.linalg.norm(projected)
                distances = (polygon - midpoint) @ direction + offset
                inside = distances <= 1e-10
                if inside.all():
                    continue
                if not inside.any():
                    polygon = np.empty((0, 3))
                    break
                clipped = []
                for j in range(len(polygon)):
                    k = (j + 1) % len(polygon)
                    if inside[j]:
                        clipped.append(polygon[j])
                    if inside[j] != inside[k]:
                        fraction = distances[j] / (distances[j] - distances[k])
                        clipped.append(polygon[j] + fraction * (polygon[k] - polygon[j]))
                polygon = np.asarray(clipped)
            if len(polygon) < 3:
                continue
            fragments.setdefault(index, []).append(polygon)
    from soundform.geometry.voronoi_filters import reach_planes, clip_plane, clip_radius_chords
    vertices, faces, labels = [], [], []
    for index, polygons in fragments.items():
        needs_cut = max_reach > 0 and any(np.linalg.norm(poly-sites[index], axis=1).max() > max_reach for poly in polygons)
        planes = reach_planes(polygons, sites[index], max_reach) if needs_cut else []
        for polygon in polygons:
            for normal, offset in planes:
                polygon = clip_plane(polygon, normal, offset)
                if len(polygon) < 3:
                    break
            if needs_cut and len(polygon) >= 3:
                polygon = clip_radius_chords(polygon, sites[index], max_reach)
            if len(polygon) < 3:
                continue
            start = len(vertices)
            vertices.extend(polygon)
            for j in range(1, len(polygon)-1):
                if np.linalg.norm(np.cross(polygon[j]-polygon[0], polygon[j+1]-polygon[0])) < 1e-12:
                    continue
                faces.extend((3, start, start+j, start+j+1))
                labels.append(index)
    if not faces:
        raise ValueError("No Voronoi panels remain. Increase maximum reach, reduce the cell gap, or use more points.")
    mesh = pv.PolyData(np.asarray(vertices), np.asarray(faces))
    mesh.cell_data["voronoi_cell"] = np.asarray(labels)
    ids = source[np.asarray(labels, dtype=int)]
    mesh.cell_data["center_source_point"] = ids
    mesh.cell_data["center_energy"] = (np.asarray(energy)[ids] if energy is not None else np.zeros(len(ids)))
    return mesh.clean()
