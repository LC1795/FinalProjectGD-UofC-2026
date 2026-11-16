"""Build thick wall meshes with through-openings and a directional height profile."""

import numpy as np

from soundform.geometry.mesh_utils import as_mesh


def wall_mesh(path, height, solid):
    """Connect two wall faces with caps along both the perimeter and the holes."""
    count, levels = height.shape
    xy = np.broadcast_to(path[:, :, None, :], (2, count, levels, 2))
    vertices = np.concatenate((xy, np.broadcast_to(height, (2, count, levels))[..., None]), axis=-1)
    indices = np.arange(2 * count * levels).reshape(2, count, levels)
    faces = []

    def quad(a, b, c, d):
        faces.extend(((a, b, c), (a, c, d)))

    for i, j in np.argwhere(solid):
        side = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        front, back = [[indices[k, x, y] for x, y in side] for k in (0, 1)]
        quad(*front)
        quad(*reversed(back))
        for edge, (x, y) in enumerate(((i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j))):
            if not (0 <= x < solid.shape[0] and 0 <= y < solid.shape[1] and solid[x, y]):
                next_edge = (edge + 1) % 4
                quad(front[next_edge], front[edge], back[edge], back[next_edge])
    mesh = as_mesh(vertices.reshape(-1, 3), np.asarray(faces))
    mesh.fix_normals()
    return mesh


def wall_grid(length, perforation, entrances=()):
    """Return a panel grid with real voids through its upper middle band."""
    bays = max(1, int(np.ceil(length)))
    edges = np.linspace(0, 1, bays + 1)
    width = perforation / 2
    if width:
        mids = (edges[1:] + edges[:-1]) / 2
        base = edges.copy()
        edges = np.unique(np.r_[edges, mids - width / bays, mids + width / bays])
    else:
        base = edges
    for entrance in entrances:
        edges = np.unique(np.r_[edges, [limit for limit in entrance if 0 < limit < 1]])
    stops = (edges[1:] + edges[:-1]) / 2
    solid = np.ones((len(stops), 3), dtype=bool)
    if width:
        bay = np.searchsorted(base, stops, side="right") - 1
        fraction = (stops - base[bay]) * bays
        solid[:, 1] = np.abs(fraction - 0.5) >= width - 1e-9
    for start, end in entrances:
        solid[(stops > start) & (stops < end)] = False
    return edges, solid
