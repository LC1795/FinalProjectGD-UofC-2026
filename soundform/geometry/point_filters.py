"""Geometric filtering of surface samples, independent of sound intensity."""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def subset_points(points, keep):
    result = pv.PolyData(np.asarray(points.points)[keep].copy())
    for name in points.point_data:
        result.point_data[name] = np.asarray(points.point_data[name])[keep].copy()
    return result


def merge_close_points(points, distance):
    """Keep one existing surface sample per radius neighbourhood, in input order.

    Avoid transitive chain collapse and preserve the surface and sample metadata.
    Remaining representatives are farther apart than the merge distance.
    """
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Merging distance must be nonnegative.")
    if distance == 0 or points.n_points < 2:
        return points.copy()
    tree = cKDTree(points.points)
    removed = np.zeros(points.n_points, bool)
    keep = []
    for i in range(points.n_points):
        if removed[i]:
            continue
        keep.append(i)
        removed[tree.query_ball_point(points.points[i], distance)] = True
    return subset_points(points, np.asarray(keep, dtype=int))


def subtract_curve_tubes(points, curves):
    """Remove samples within radius of finite polyline segments (rounded ends).

    curves contains (vertices, packed VTK lines, radius); all units are metres.
    """
    xyz = np.asarray(points.points)
    removed = np.zeros(points.n_points, bool)
    if not len(xyz):
        return points.copy(), 0
    tree = cKDTree(xyz)
    for vertices, lines, radius in curves:
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Circulation radius must be nonnegative.")
        if radius == 0:
            continue
        vertices = np.asarray(vertices)
        offset = 0
        while offset < len(lines):
            count = int(lines[offset])
            ids = lines[offset+1:offset+1+count]
            offset += count+1
            for i, j in zip(ids[:-1], ids[1:]):
                a, b = vertices[i], vertices[j]
                delta = b-a
                length = np.linalg.norm(delta)
                ids_near = np.asarray(tree.query_ball_point((a+b)/2, radius+length/2), dtype=int)
                if not len(ids_near):
                    continue
                t = np.clip((xyz[ids_near]-a)@delta/max(length*length, 1e-30), 0, 1)
                distance = np.linalg.norm(xyz[ids_near]-(a+t[:,None]*delta), axis=1)
                removed[ids_near[distance <= radius]] = True
    return subset_points(points, ~removed), int(removed.sum())
