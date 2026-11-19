"""Delaunay and uniform-grid Poisson reconstruction of oriented cloud samples."""

import numpy as np
import pyvista as pv
from scipy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


def reconstruct_cloud(points, method, alpha=0.0, resolution=64, boundary=None, voronoi_cells=250, voronoi_gap=0.08, voronoi_seed=42):
    if points is None or points.n_points < 12:
        raise ValueError("Generate at least 12 cloud surface points first.")
    xyz = np.asarray(points.points)
    if not np.isfinite(xyz).all():
        raise ValueError("Cloud point coordinates must be finite.")
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 3:
        raise ValueError("The points must span a three-dimensional surface.")
    if method == "delaunay":
        mesh = pv.PolyData(xyz).delaunay_3d(alpha=alpha).extract_surface(algorithm="dataset_surface").triangulate()
        # Alpha shapes can contain isolated lines; retain only triangular faces.
        mesh = pv.PolyData(mesh.points, mesh.faces).clean()
    elif method == "voronoi":
        mesh = _voronoi(xyz, boundary, voronoi_cells, voronoi_gap, voronoi_seed)
    elif method == "poisson":
        if "Normals" not in points.point_data:
            raise ValueError("Regenerate the surface points to include their oriented normals.")
        mesh = _poisson(xyz, np.asarray(points.point_data["Normals"]), resolution)
    else:
        raise ValueError("Unknown reconstruction method.")
    if not mesh.n_cells:
        raise ValueError("No surface was reconstructed. Adjust alpha or generate more points.")
    _, nearest = cKDTree(xyz).query(mesh.points)
    mesh.point_data["energy"] = np.asarray(points.point_data["energy"])[nearest]
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


def _voronoi(points, boundary, cell_count=250, gap=0.08, seed=42):
    """Restricted Euclidean Voronoi patches on the actual triangulated surface.

    Each surface triangle is intersected with nearest-site half-spaces. No
    volumetric walls, projection to a plane, or closure across the terrain rim.
    """
    if boundary is None or not boundary.n_cells:
        raise ValueError("Create a cloud before generating its Voronoi mesh.")
    if not np.isfinite(gap) or gap < 0 or not 0 <= cell_count <= 5000:
        raise ValueError("Use a nonnegative gap and 0-5000 cells (0 uses all points).")
    sites = np.unique(points, axis=0)
    if cell_count and cell_count < len(sites):
        rng = np.random.default_rng(seed)
        sites = sites[np.sort(rng.choice(len(sites), int(cell_count), replace=False))]
    tree = cKDTree(sites)
    surface = boundary.triangulate()
    triangles = surface.points[surface.faces.reshape(-1, 4)[:, 1:]]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    nearest, _ = tree.query(centers)
    # A winner anywhere on a triangle is at most nearest(center)+2*radius
    # from its center. Include extra margin for inset cell borders.
    candidates = tree.query_ball_point(centers, nearest + 2 * radii + 2 * gap + 1e-9)
    vertices, faces, labels = [], [], []
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
            start = len(vertices)
            vertices.extend(polygon)
            for j in range(1, len(polygon) - 1):
                if np.linalg.norm(np.cross(polygon[j] - polygon[0], polygon[j+1] - polygon[0])) < 1e-12:
                    continue
                faces.extend((3, start, start + j, start + j + 1))
                labels.append(index)
    if not faces:
        raise ValueError("No Voronoi panels remain. Reduce the cell gap or the number of cells.")
    mesh = pv.PolyData(np.asarray(vertices), np.asarray(faces))
    mesh.cell_data["voronoi_cell"] = np.asarray(labels)
    return mesh.clean()
