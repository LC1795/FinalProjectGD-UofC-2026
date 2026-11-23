"""Surface routing and spatial intersection checks for cloud line networks."""
import itertools
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def surface_candidates(xyz, surface, max_distance):
    if surface is None or not surface.n_cells:
        raise ValueError("Create a metaball surface before enabling surface connections.")
    mesh = surface.triangulate().clean()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    n = mesh.n_points
    cells = np.asarray(mesh.find_closest_cell(xyz))
    coordinates = np.vstack((mesh.points, xyz))
    edges = [faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]
    for corner in range(3):
        edges.append(np.c_[np.arange(len(xyz)) + n, faces[cells, corner]])
    # Direct links within one surface triangle avoid coarse-mesh detours.
    tree = cKDTree(xyz)
    _, ids = tree.query(xyz, k=min(17, len(xyz)), distance_upper_bound=max_distance)
    first = np.repeat(np.arange(len(xyz)), ids.shape[1])
    second = ids.ravel()
    valid = (second < len(xyz)) & (first != second)
    first, second = first[valid], second[valid]
    valid = cells[first] == cells[second]
    edges.append(np.c_[first[valid]+n, second[valid]+n])
    edges = np.unique(np.sort(np.vstack(edges), axis=1), axis=0)
    weights = np.maximum(np.linalg.norm(coordinates[edges[:,0]] - coordinates[edges[:,1]], axis=1), 1e-12)
    graph = coo_matrix((np.r_[weights,weights],
                       (np.r_[edges[:,0],edges[:,1]], np.r_[edges[:,1],edges[:,0]])),
                      shape=(len(coordinates),len(coordinates))).tocsr()
    distances, previous, owners = dijkstra(graph, directed=True, indices=np.arange(n,len(coordinates)),
                                          min_only=True, return_predecessors=True, limit=max_distance)
    winners = {}
    a, b = edges.T
    valid = (owners[a] >= n) & (owners[b] >= n) & (owners[a] != owners[b])
    for a,b,weight in zip(a[valid], b[valid], weights[valid]):
        length = float(distances[a] + weight + distances[b])
        if length > max_distance:
            continue
        i,j = int(owners[a]-n),int(owners[b]-n)
        key = (min(i,j),max(i,j))
        if key not in winners or length < winners[key][0]:
            winners[key] = (length,int(a),int(b))
    def trace(node):
        path = [node]
        while previous[node] != -9999:
            node = int(previous[node])
            path.append(node)
        return path
    for (i,j),(length,a,b) in sorted(winners.items(), key=lambda item:item[1][0]):
        nodes = trace(a)[::-1] + trace(b)
        route = coordinates[nodes]
        if int(owners[a]-n) != i:
            route = route[::-1]
        keep = np.r_[True,np.linalg.norm(np.diff(route,axis=0),axis=1)>1e-12]
        yield i,j,length,route[keep]


def _segment_distances(p, end, starts, ends):
    """Minimum distances to finite segments, including parallel/collinear cases."""
    u = end-p
    v = ends-starts
    w = p-starts
    a = float(u@u)
    c = np.einsum("ij,ij->i",v,v)
    def point_distance(point, base, vector, norm):
        t = np.clip(np.einsum("ij,ij->i",point-base,vector)/np.maximum(norm,1e-30),0,1)
        return np.linalg.norm(point-(base+t[:,None]*vector),axis=1)
    broadcast_u = np.broadcast_to(u,starts.shape)
    broadcast_p = np.broadcast_to(p,starts.shape)
    result = np.minimum.reduce([
        point_distance(broadcast_p,starts,v,c),
        point_distance(np.broadcast_to(end,starts.shape),starts,v,c),
        point_distance(starts,broadcast_p,broadcast_u,np.full(len(starts),a)),
        point_distance(ends,broadcast_p,broadcast_u,np.full(len(starts),a))])
    b = v@u
    d = w@u
    e = np.einsum("ij,ij->i",v,w)
    denominator = a*c-b*b
    safe = np.where(denominator>1e-20,denominator,1)
    s = (b*e-c*d)/safe
    t = (a*e-b*d)/safe
    valid = (denominator>1e-20)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    interior = np.linalg.norm(w+s[:,None]*u-t[:,None]*v,axis=1)
    return np.where(valid,np.minimum(result,interior),result)


class CrossingIndex:
    def __init__(self, xyz, cell_size, tolerance):
        self.xyz = xyz
        self.cell_size = max(cell_size, tolerance*4)
        self.tolerance = tolerance
        self.bins = {}
        self.segments = []
        self.pairs = []

    def _bins(self, start, end):
        low = np.floor((np.minimum(start,end)-self.tolerance)/self.cell_size).astype(int)
        high = np.floor((np.maximum(start,end)+self.tolerance)/self.cell_size).astype(int)
        return itertools.product(*(range(a,b+1) for a,b in zip(low,high)))

    def crosses(self, route, pair):
        for p,q in zip(route[:-1],route[1:]):
            nearby = set()
            for key in self._bins(p,q):
                nearby.update(self.bins.get(key,()))
            if not nearby:
                continue
            indices = sorted(nearby)
            segments = np.asarray([self.segments[k] for k in indices])
            # Trim only legitimate shared cloud endpoints. Interior crossings and
            # collinear overlaps remain collisions, including on surface routes.
            for k,identifier in enumerate(indices):
                a,b = p.copy(),q.copy()
                c,d = segments[k].copy()
                for common in set(pair).intersection(self.pairs[identifier]):
                    point = self.xyz[common]
                    for segment in ((a,b),(c,d)):
                        delta = segment[1]-segment[0]
                        length = np.linalg.norm(delta)
                        if length <= 1e-12:
                            continue
                        trim = min(self.tolerance*2.1,length*.4)*delta/length
                        if np.linalg.norm(segment[0]-point)<1e-9:
                            segment[0][:] += trim
                        if np.linalg.norm(segment[1]-point)<1e-9:
                            segment[1][:] -= trim
                if _segment_distances(a,b,c[None],d[None])[0] < self.tolerance:
                    return True
        return False

    def add(self, route, pair):
        for start,end in zip(route[:-1],route[1:]):
            identifier = len(self.segments)
            self.segments.append((start.copy(),end.copy()))
            self.pairs.append(pair)
            for key in self._bins(start,end):
                self.bins.setdefault(key,[]).append(identifier)
