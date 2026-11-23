"""Straight reach closures and conservative circulation cutouts."""
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, QhullError


def clip_plane(polygon, normal, offset):
    if len(polygon) < 3:
        return np.empty((0, 3))
    ds = polygon @ normal-offset
    inside = ds <= 1e-10
    if inside.all():
        return polygon
    if not inside.any():
        return np.empty((0, 3))
    result = []
    for j in range(len(polygon)):
        k = (j+1) % len(polygon)
        if inside[j]:
            result.append(polygon[j])
        if inside[j] != inside[k]:
            result.append(polygon[j]+ds[j]/(ds[j]-ds[k])*(polygon[k]-polygon[j]))
    return np.asarray(result)


def reach_planes(polygons, center, radius):
    """Join whole-cell boundary/radius crossings, not each triangle's arc.

    Project those endpoints to a local plane and use their convex footprint
    as cutting planes. Intersect with the original surface afterwards.
    """
    vertices, faces = [], []
    for polygon in polygons:
        start = len(vertices)
        vertices.extend(polygon)
        faces.extend([len(polygon), *range(start, start+len(polygon))])
    mesh = pv.PolyData(np.asarray(vertices), np.asarray(faces)).clean(tolerance=1e-8, absolute=True)
    boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False)
    # Local surface orientation near the generating sample.
    tri = mesh.triangulate()
    face = tri.faces.reshape(-1, 4)[tri.find_closest_cell(center), 1:]
    corner = tri.points[face]
    normal = np.cross(corner[1]-corner[0], corner[2]-corner[0])
    normal /= max(np.linalg.norm(normal), 1e-30)
    u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    points = []
    if boundary.n_points:
        xyz = np.asarray(boundary.points)
        points.extend(xyz[np.linalg.norm(xyz-center, axis=1) <= radius])
        # Feature edges are individual line segments.
        for a, b in xyz[boundary.lines.reshape(-1, 3)[:, 1:]]:
            delta, rel = b-a, a-center
            aa = float(delta@delta)
            bb = float(2*rel@delta)
            cc = float(rel@rel-radius*radius)
            disc = bb*bb-4*aa*cc
            if aa <= 1e-24 or disc < 0:
                continue
            for t in ((-bb-np.sqrt(disc))/(2*aa), (-bb+np.sqrt(disc))/(2*aa)):
                if 0 <= t <= 1:
                    points.append(a+t*delta)
    if len(points) >= 3:
        uv = (np.asarray(points)-center) @ np.array([u, v]).T
        uv = np.unique(uv, axis=0)
        try:
            hull = ConvexHull(uv)
            return [(row[0]*u+row[1]*v,
                     float((row[0]*u+row[1]*v)@center-row[2])) for row in hull.equations]
        except QhullError:
            pass
    # When the radius sits wholly inside a cell there are no cut endpoints.
    # Use a square footprint, rather than inventing a circular tile edge.
    reach = radius/np.sqrt(2)
    return [(direction, float(direction@center+reach)) for direction in (u, -u, v, -v)]


def clip_radius_chords(polygon, center, radius):
    """Keep inside vertices and exact edge/radius intersections, with straight joins."""
    points = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if np.linalg.norm(a-center) <= radius+1e-10:
            points.append(a)
        delta, rel = b-a, a-center
        aa = float(delta@delta)
        bb = float(2*rel@delta)
        cc = float(rel@rel-radius*radius)
        disc = bb*bb-4*aa*cc
        if aa > 1e-24 and disc > 0:
            roots = sorted([(-bb-np.sqrt(disc))/(2*aa), (-bb+np.sqrt(disc))/(2*aa)])
            points.extend(a+t*delta for t in roots if 1e-10 < t < 1-1e-10)
    return np.asarray(points) if len(points) >= 3 else np.empty((0, 3))


def _collinear_vertices(vertices):
    """Remove redundant straight-run samples without simplifying curved paths."""
    kept = []
    for point in np.asarray(vertices, dtype=float):
        if kept and np.linalg.norm(point-kept[-1]) <= 1e-12:
            continue
        while len(kept) >= 2:
            a, b = kept[-2], kept[-1]
            delta = point-a
            length = np.linalg.norm(delta)
            if length <= 1e-12 or (b-a)@(point-b) < 0:
                break
            distance = np.linalg.norm(np.cross(b-a,delta))/length
            if distance > 1e-10:
                break
            kept.pop()
        kept.append(point)
    return np.asarray(kept)


def _circulation_cutters(curves):
    sphere = pv.Icosphere(nsub=1).points.astype(float)
    sphere /= np.linalg.norm(sphere,axis=1)[:,None]
    angles = np.arange(32)*2*np.pi/32
    for vertices, lines, radius in curves:
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Circulation radius must be nonnegative.")
        if radius == 0:
            continue
        offset = 0
        while offset < len(lines):
            count = int(lines[offset])
            ids = lines[offset+1:offset+1+count]
            offset += count+1
            polyline = _collinear_vertices(np.asarray(vertices)[ids])
            pairs = list(zip(polyline[:-1],polyline[1:]))
            # A collapsed zero-length polyline still represents a round clearance.
            if len(polyline) == 1:
                pairs = [(polyline[0],polyline[0])]
            for a,b in pairs:
                axis = b-a
                length = np.linalg.norm(axis)
                if length > 1e-12:
                    axis /= length
                    u = np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))])
                    u /= np.linalg.norm(u)
                    v = np.cross(axis,u)
                    directions = np.vstack([sphere,axis,-axis,
                        np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v])
                else:
                    directions = sphere
                limits = np.maximum(directions@a,directions@b)+radius
                yield directions, limits, np.minimum(a,b)-radius*1.1, np.maximum(a,b)+radius*1.1, (a+b)/2, length/2+radius*1.1


def _subtract_convex(polygon, directions, limits):
    """Subtract one capsule from one convex fragment, without triangulating."""
    ds = polygon@directions.T-limits
    if (ds <= 1e-10).all():
        return []
    if (ds >= -1e-10).all(axis=0).any():
        return [polygon]
    # Planes containing the entire polygon cannot cut any of its fragments.
    cutting = np.flatnonzero((ds > 1e-10).any(axis=0))
    output = []
    remaining = polygon
    for k in cutting:
        normal, limit = directions[k], limits[k]
        values = remaining@normal-limit
        if (values <= 1e-10).all():
            continue
        if (values >= -1e-10).all():
            output.append(remaining)
            return output
        outside = clip_plane(remaining,-normal,-limit)
        if len(outside) >= 3:
            output.append(outside)
        remaining = clip_plane(remaining,normal,limit)
        if len(remaining) < 3:
            break
    return output


def subtract_circulation(mesh, curves):
    """Batch pipe subtraction against original triangles; assemble only once.

    A spatial tree limits cutters to nearby triangles. Convex fragments remain
    polygons until every relevant pipe has been applied, avoiding repeated
    triangulation, full-mesh rebuilding, and multiplication of internal edges.
    Capsule planes and the conservative pipe clearances are unchanged.
    """
    from scipy.spatial import cKDTree
    if not mesh.n_cells or not curves:
        return mesh.copy()
    triangles = np.asarray(mesh.points,dtype=float)[mesh.faces.reshape(-1,4)[:,1:]]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles-centers[:,None],axis=2).max(axis=1)
    low, high = triangles.min(axis=1),triangles.max(axis=1)
    tree = cKDTree(centers)
    largest = float(radii.max())
    cutters, candidates = [], {}
    for directions,limits,lo,hi,center,radius in _circulation_cutters(curves):
        ids = np.asarray(tree.query_ball_point(center,radius+largest),dtype=int)
        if not len(ids):
            continue
        valid = (high[ids]>=lo).all(axis=1)&(low[ids]<=hi).all(axis=1)
        ids = ids[valid]
        if not len(ids):
            continue
        # Vectorized separating-plane tests remove false AABB candidates.
        nearby = []
        for start in range(0,len(ids),512):
            batch = ids[start:start+512]
            ds = triangles[batch]@directions.T-limits
            nearby.extend(batch[~(ds>=-1e-10).all(axis=1).any(axis=1)])
        ids = np.asarray(nearby,dtype=int)
        if not len(ids):
            continue
        key = len(cutters)
        cutters.append((directions,limits))
        for index in ids:
            candidates.setdefault(int(index),[]).append(key)
    if not candidates:
        return mesh.copy()
    # Untouched triangles are copied in bulk, retaining their original vertices.
    touched = np.zeros(len(triangles),bool)
    touched[list(candidates)] = True
    owners = list(np.flatnonzero(~touched))
    faces = list(mesh.faces.reshape(-1,4)[~touched].ravel())
    points = list(np.asarray(mesh.points,dtype=float))
    for index, keys in candidates.items():
        fragments = [triangles[index]]
        for key in keys:
            directions,limits = cutters[key]
            fragments = [piece for poly in fragments for piece in _subtract_convex(poly,directions,limits)]
            if not fragments:
                break
        for polygon in fragments:
            start = len(points)
            points.extend(polygon)
            for k in range(1,len(polygon)-1):
                if np.linalg.norm(np.cross(polygon[k]-polygon[0],polygon[k+1]-polygon[0])) > 1e-12:
                    faces.extend([3,start,start+k,start+k+1])
                    owners.append(index)
    result = pv.PolyData(np.asarray(points).reshape(-1,3),np.asarray(faces,dtype=np.int64))
    for name in mesh.cell_data:
        result.cell_data[name] = np.asarray(mesh.cell_data[name])[np.asarray(owners,dtype=int)]
    for name in mesh.field_data:
        result.field_data[name] = mesh.field_data[name]
    # Empty output must not become a point cloud via VTK's default vertex cells.
    result.verts = np.empty(0,dtype=np.int64)
    return result.clean()
