"""Approximate intrinsic Delaunay network from geodesic Voronoi regions.

The existing surface supplies topology; only sample positions and the distance
cap determine the network. No audio scalars or sample normals are consulted.
"""
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def triangulate_surface_points(points, surface, max_distance=0.0):
    if surface is None or not surface.n_cells:
        raise ValueError("Generate the metaball surface first.")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("Distance cap must be nonnegative; zero means unlimited.")
    xyz, source = np.unique(np.asarray(points, dtype=float), axis=0, return_index=True)
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("Use at least three distinct finite surface points.")
    mesh = surface.triangulate().clean()
    mesh.clear_data()
    spacing = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    target = float(np.median(spacing)) * .5
    # Bounded refinement resolves Voronoi meetings without an all-pairs matrix.
    for _ in range(3):
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        longest = np.linalg.norm(mesh.points[faces[:, 0]]-mesh.points[faces[:, 1]], axis=1)
        if np.median(longest) <= target or mesh.n_cells * 4 > 300000:
            break
        mesh = mesh.subdivide(1, subfilter="linear")
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    cells = np.asarray(mesh.find_closest_cell(xyz))
    n = mesh.n_points
    coordinates = np.vstack([mesh.points, xyz])
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                       *[np.c_[np.arange(len(xyz))+n, faces[cells, k]] for k in range(3)]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    weights = np.maximum(np.linalg.norm(coordinates[edges[:, 0]]-coordinates[edges[:, 1]], axis=1), 1e-12)
    graph = coo_matrix((np.r_[weights, weights],
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(coordinates),)*2).tocsr()
    distances, _, owners = dijkstra(graph, indices=np.arange(n, n+len(xyz)),
                                    min_only=True, return_predecessors=True)
    labels = owners[:n] - n
    triples = labels[faces]
    valid = (triples >= 0).all(axis=1) & (triples[:, 0] != triples[:, 1]) & (triples[:, 1] != triples[:, 2]) & (triples[:, 2] != triples[:, 0])
    face_ids = np.flatnonzero(valid)
    triples = triples[valid]
    # Orient each dual triangle using its reference surface face.
    ref = mesh.points[faces[face_ids]]
    normals = np.cross(ref[:, 1]-ref[:, 0], ref[:, 2]-ref[:, 0])
    corners = xyz[triples]
    reverse = np.einsum("ij,ij->i", np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]), normals) < 0
    triples[reverse] = triples[reverse][:, [0, 2, 1]]
    _, unique = np.unique(np.sort(triples, axis=1), axis=0, return_index=True)
    triples = triples[np.sort(unique)]
    # A meeting-boundary route provides a conservative surface distance estimate.
    route_lengths = {}
    for (a, b), weight in zip(edges, weights):
        i, j = int(owners[a]-n), int(owners[b]-n)
        if i < 0 or j < 0 or i == j:
            continue
        key = (min(i, j), max(i, j))
        length = float(distances[a]+weight+distances[b])
        route_lengths[key] = min(route_lengths.get(key, np.inf), length)
    kept, lengths, incidence = [], [], {}
    for tri in triples:
        pairs = [tuple(sorted((int(tri[k]), int(tri[(k+1)%3])))) for k in range(3)]
        ds = [route_lengths.get(pair, np.inf) for pair in pairs]
        if max_distance and max(ds) > max_distance:
            continue
        corners = xyz[tri]
        if np.linalg.norm(np.cross(corners[1]-corners[0], corners[2]-corners[0])) < 1e-12:
            continue
        if any(incidence.get(pair, 0) >= 2 for pair in pairs):
            continue
        for pair in pairs:
            incidence[pair] = incidence.get(pair, 0)+1
        kept.append(tri)
        lengths.append(max(ds))
    if not kept:
        raise ValueError("No surface triangles remain. Increase the distance cap or generate more points.")
    used, inverse = np.unique(kept, return_inverse=True)
    result = pv.PolyData(xyz[used], np.c_[np.full(len(kept), 3), inverse.reshape(-1, 3)].ravel())
    result.point_data["source_point"] = source[used]
    result.cell_data["max_surface_edge_m"] = np.asarray(lengths)
    result.field_data["method"] = np.array(["geodesic_voronoi_dual"])
    return result
