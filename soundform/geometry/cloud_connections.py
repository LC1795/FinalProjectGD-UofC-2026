"""Bounded nearest-neighbor line networks from existing cloud points."""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def connect_cloud_points(points, max_distance=1.0, min_distance=0.0,
                         max_connections=8, max_lines=100000, *, surface=None, surface_only=False,
                         adaptive=False, spacing_factor=2.5, reject_crossings=False, crossing_tolerance=0.005):
    if points is None or points.n_points < 2:
        raise ValueError("Generate at least two cloud points first.")
    if (not np.isfinite([min_distance, max_distance]).all()
            or not 0 <= min_distance < max_distance):
        raise ValueError("Minimum distance must be nonnegative and smaller than maximum distance.")
    if not 1 <= max_connections <= 64 or not 1 <= max_lines <= 500000:
        raise ValueError("Use 1-64 connections per point and 1-500000 total lines.")
    xyz = np.asarray(points.points)
    if not np.isfinite(xyz).all():
        raise ValueError("Cloud points must have finite coordinates.")
    if surface_only or adaptive or reject_crossings:
        return _checked_connections(points, max_distance, min_distance, max_connections, max_lines,
                                    surface, surface_only, adaptive, spacing_factor, reject_crossings, crossing_tolerance)
    tree = cKDTree(xyz)
    count = len(xyz)
    neighbors = min(count, 4 * int(max_connections) + 1)
    degree = np.zeros(count, dtype=np.int32)
    edges, lengths, seen = [], [], set()
    capped = False
    # Batch queries bound memory even for 50,000 points and generous thresholds.
    for start in range(0, count, 512):
        distances, indices = tree.query(xyz[start:start+512], k=neighbors,
                                       distance_upper_bound=np.nextafter(max_distance, np.inf))
        for row, (ds, ids) in enumerate(zip(distances, indices)):
            first = start + row
            if degree[first] >= max_connections:
                continue
            for distance, second in zip(ds, ids):
                second = int(second)
                if second >= count or second == first or not np.isfinite(distance):
                    continue
                if distance <= 1e-12 or distance < min_distance or distance > max_distance:
                    continue
                if degree[second] >= max_connections:
                    continue
                pair = (min(first, second), max(first, second))
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append(pair)
                lengths.append(distance)
                degree[first] += 1
                degree[second] += 1
                if len(edges) >= max_lines:
                    capped = True
                    break
                if degree[first] >= max_connections:
                    break
            if capped:
                break
        if capped:
            break
    if not edges:
        raise ValueError("No point pairs meet these distances. Increase maximum distance or lower minimum distance.")
    edges = np.asarray(edges, dtype=np.int64)
    used, inverse = np.unique(edges.ravel(), return_inverse=True)
    mesh = pv.PolyData(xyz[used])
    mesh.verts = np.empty(0, dtype=np.int64)
    mesh.lines = np.c_[np.full(len(edges), 2), inverse.reshape(-1, 2)].ravel()
    mesh.point_data["source_point"] = used
    mesh.point_data["energy"] = (np.asarray(points.point_data["energy"])[used]
                                  if "energy" in points.point_data else np.zeros(len(used)))
    mesh.cell_data["length_m"] = np.asarray(lengths)
    mesh.field_data["line_limit_reached"] = np.array([capped], dtype=np.uint8)
    return mesh


def _checked_connections(points, maximum, minimum, degree_limit, line_limit, surface,
                         surface_only, adaptive, factor, reject_crossings, tolerance):
    from soundform.geometry.connection_checks import surface_candidates, CrossingIndex
    if not np.isfinite([factor,tolerance]).all() or factor <= 0 or tolerance <= 0:
        raise ValueError("Spacing multiplier and crossing tolerance must be positive.")
    xyz = np.asarray(points.points)
    tree = cKDTree(xyz)
    distances,_ = tree.query(xyz,k=min(7,len(xyz)))
    spacing = np.median(distances[:,1:],axis=1)
    spacing = np.maximum(spacing,1e-9)
    if surface_only:
        candidates = surface_candidates(xyz,surface,maximum)
    else:
        def raw():
            for start in range(0,len(xyz),512):
                ds,ids = tree.query(xyz[start:start+512],k=min(len(xyz),4*degree_limit+1),distance_upper_bound=maximum)
                for row,(values,neighbors) in enumerate(zip(ds,ids)):
                    i = start+row
                    for distance,j in zip(values,neighbors):
                        if j < len(xyz) and j > i:
                            yield i,int(j),float(distance),xyz[[i,j]]
        candidates = raw()
    crossing = CrossingIndex(xyz,min(maximum,float(np.median(spacing))*2),tolerance)
    degree = np.zeros(len(xyz),int)
    routes, pairs, lengths = [], [], []
    seen = set()
    rejected = dict(distance=0, spacing=0, crossing=0)
    for i,j,length,route in candidates:
        pair = (min(i,j),max(i,j))
        if pair in seen or degree[i]>=degree_limit or degree[j]>=degree_limit:
            continue
        seen.add(pair)
        if length <= 1e-12 or length < minimum or length > maximum:
            rejected["distance"] += 1
            continue
        if adaptive and length > factor*min(spacing[i],spacing[j]):
            rejected["spacing"] += 1
            continue
        if reject_crossings and crossing.crosses(route,pair):
            rejected["crossing"] += 1
            continue
        routes.append(route)
        pairs.append((i,j))
        lengths.append(length)
        degree[i]+=1
        degree[j]+=1
        if reject_crossings:
            crossing.add(route,pair)
        if len(routes)>=line_limit:
            break
    if not routes:
        raise ValueError("No connections remain after the checks. Increase the distance or spacing multiplier, or relax crossing checks.")
    vertices, lines, energies, source_ids = [], [], [], []
    energy = np.asarray(points["energy"]) if "energy" in points.point_data else np.zeros(len(xyz))
    for route,(i,j) in zip(routes,pairs):
        start = len(vertices)
        vertices.extend(route)
        lines.extend([len(route),*range(start,start+len(route))])
        arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        energies.extend(energy[i]+(energy[j]-energy[i])*arc/max(arc[-1],1e-12))
        source_ids.extend([i,*([-1]*(len(route)-2)),j])
    mesh = pv.PolyData(np.asarray(vertices))
    mesh.verts = np.empty(0,dtype=np.int64)
    mesh.lines = np.asarray(lines,dtype=np.int64)
    mesh.point_data["energy"] = np.asarray(energies)
    mesh.point_data["source_point"] = np.asarray(source_ids)
    mesh.cell_data["length_m"] = np.asarray(lengths)
    mesh.cell_data["start_point"] = np.array(pairs)[:,0]
    mesh.cell_data["end_point"] = np.array(pairs)[:,1]
    mesh.field_data["line_limit_reached"] = np.array([len(routes)>=line_limit],dtype=np.uint8)
    for name,value in rejected.items():
        mesh.field_data["rejected_"+name] = np.array([value])
    return mesh
