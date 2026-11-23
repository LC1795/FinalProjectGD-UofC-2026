"""Strut/node meshes and closed Voronoi tile shells."""
import numpy as np
import pyvista as pv


def _range(minimum, maximum):
    if not np.isfinite([minimum,maximum]).all() or not 0 < minimum <= maximum:
        raise ValueError("Use positive minimum and maximum sizes, with minimum no greater than maximum.")


def _weights(cloud, ids, required):
    if "energy" not in cloud.point_data:
        if required:
            raise ValueError("The source points have no sound weights. Regenerate points after audio analysis.")
        return np.zeros(len(ids))
    energy = np.asarray(cloud.point_data["energy"], dtype=float)[ids]
    if not np.isfinite(energy).all():
        raise ValueError("Point sound weights must be finite.")
    return np.clip(energy,0,1)


def delaunay_frame(mesh, cloud, radius=.05, weighted=False, min_radius=.02, max_radius=.15):
    """Unique capped struts and surface-point spheres; weighted struts taper."""
    _range(min_radius,max_radius) if weighted else _range(radius,radius)
    if not mesh.n_cells or "source_point" not in mesh.point_data or cloud is None:
        raise ValueError("Generate the Delaunay mesh first.")
    xyz = np.asarray(mesh.points,dtype=float)
    faces = mesh.faces.reshape(-1,4)[:,1:]
    edges = np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    ids = np.asarray(mesh.point_data["source_point"],dtype=int)
    if np.any(ids<0) or np.any(ids>=cloud.n_points):
        raise ValueError("Delaunay source points changed. Regenerate the mesh.")
    energy = _weights(cloud,ids,weighted)
    radii = min_radius+(max_radius-min_radius)*energy if weighted else np.full(len(xyz),radius)
    used = np.unique(edges)
    sphere = pv.Sphere(radius=1,theta_resolution=12,phi_resolution=10)
    sphere_faces = sphere.faces.reshape(-1,4)[:,1:]
    sides = 12
    estimated = len(edges)*sides*4+len(used)*len(sphere_faces)
    if estimated > 4000000:
        raise ValueError("This frame would exceed four million triangles. Generate fewer points before thickening.")
    vertices, triangles, values, kinds, owners = [], [], [], [], []
    offset = 0
    angles = np.arange(sides)*2*np.pi/sides
    def append(v,f,e,kind,owner):
        nonlocal offset
        vertices.append(v)
        triangles.append(f+offset)
        values.append(e)
        kinds.extend([kind]*len(f))
        owners.extend([owner]*len(f))
        offset += len(v)
    for edge_id,(i,j) in enumerate(edges):
        delta = xyz[j]-xyz[i]
        length = np.linalg.norm(delta)
        if length <= 1e-12:
            continue
        axis = delta/length
        u = np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))])
        u /= np.linalg.norm(u)
        v = np.cross(axis,u)
        ring = np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
        points = np.vstack([xyz[i]+radii[i]*ring,xyz[j]+radii[j]*ring,xyz[i],xyz[j]])
        cells = []
        for a in range(sides):
            b = (a+1)%sides
            cells.extend([[a,b,b+sides],[a,b+sides,a+sides],
                          [2*sides,b,a],[2*sides+1,a+sides,b+sides]])
        append(points,np.asarray(cells),np.r_[np.full(sides,energy[i]),np.full(sides,energy[j]),energy[i],energy[j]],0,edge_id)
    for node in used:
        append(np.asarray(sphere.points,dtype=float)*radii[node]+xyz[node],sphere_faces,
               np.full(sphere.n_points,energy[node]),1,int(node))
    if not vertices:
        raise ValueError("No Delaunay edges remain.")
    result = pv.PolyData(np.vstack(vertices),np.c_[np.full(sum(map(len,triangles)),3),np.vstack(triangles)].ravel())
    result.point_data["energy"] = np.concatenate(values)
    result.cell_data["primitive_kind"] = np.asarray(kinds,dtype=np.uint8)
    result.cell_data["primitive_index"] = np.asarray(owners)
    result.field_data["node_source_point"] = ids[used]
    result.field_data["node_radius_m"] = radii[used]
    result.field_data["edge_source_points"] = ids[edges].ravel()
    result.field_data["sizing"] = np.array(["sound_weighted" if weighted else "uniform"])
    result.field_data["source_method"] = np.array(["delaunay"])
    return result


def _separate_vertex_fans(xyz,faces):
    """Separate touching sheets and vertex pinches without filling any holes.

    Faces on a genuine branching edge become independent small patches; the
    remaining edge-connected vertex fans retain shared topology. Each resulting
    patch can receive its own closed boundary walls during extrusion.
    """
    n=len(faces)
    edges=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    face_ids=np.tile(np.arange(n),3)
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    branching=np.zeros(n,bool)
    branching[face_ids[counts[inverse]>2]]=True
    parent=np.arange(n*3)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def join(a,b):
        a,b=root(a),root(b)
        if a!=b:
            parent[b]=a
    order=np.argsort(inverse,kind="stable")
    start=0
    for count in counts:
        group=order[start:start+count]
        start+=count
        if count!=2:
            continue
        a,b=map(int,face_ids[group])
        if branching[a] or branching[b]:
            continue
        for vertex in edges[group[0]]:
            ai=int(np.flatnonzero(faces[a]==vertex)[0])
            bi=int(np.flatnonzero(faces[b]==vertex)[0])
            join(a*3+ai,b*3+bi)
    roots=np.array([root(i) for i in range(n*3)])
    _,first,indices=np.unique(roots,return_index=True,return_inverse=True)
    points=xyz[faces.ravel()[first]]
    f=indices.reshape(-1,3)
    result=pv.PolyData(points,np.c_[np.full(n,3),f].ravel())
    result.field_data["branch_faces_separated"]=np.array([int(branching.sum())])
    return result


def _conform_tile(tile):
    """Split T-junctions left by pipe cuts before building boundary walls."""
    from scipy.spatial import cKDTree
    # Surface extraction uses float32 coordinates; a fixed 1e-8 weld was
    # below their precision and left duplicate cut vertices unmerged.
    span = float(np.ptp(tile.points,axis=0).max())
    tolerance = max(span*1e-7,1e-8)
    tile = tile.clean(tolerance=tolerance,absolute=True)
    xyz = np.asarray(tile.points,dtype=float)
    faces = tile.faces.reshape(-1,4)[:,1:]
    # Cutting and welding can leave duplicate or collapsed triangles. Remove
    # them before adding centre-fan vertices, which would hide duplicates.
    vectors = np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
    faces = faces[np.linalg.norm(vectors,axis=1)>1e-12]
    _,unique = np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
    faces = faces[np.sort(unique)]
    tree = cKDTree(xyz)
    edge_splits = {}
    points = list(xyz)
    output = []
    for face in faces:
        boundary = []
        for a,b in zip(face,np.roll(face,-1)):
            key = (min(int(a),int(b)),max(int(a),int(b)))
            if key not in edge_splits:
                i,j = key
                delta = xyz[j]-xyz[i]
                length = np.linalg.norm(delta)
                splits = []
                if length>tolerance:
                    nearby = np.asarray(tree.query_ball_point((xyz[i]+xyz[j])/2,length/2+tolerance),dtype=int)
                    t = (xyz[nearby]-xyz[i])@delta/(length*length)
                    error = np.linalg.norm(xyz[nearby]-(xyz[i]+t[:,None]*delta),axis=1)
                    valid = (t>tolerance/length)&(t<1-tolerance/length)&(error<=tolerance)
                    splits = list(nearby[valid][np.argsort(t[valid])])
                edge_splits[key] = splits
            splits = edge_splits[key]
            boundary.extend([int(a),*(splits if a<b else splits[::-1])])
        if len(boundary)==3:
            output.append(face)
        else:
            center = len(points)
            points.append(xyz[face].mean(axis=0))
            output.extend([center,int(a),int(b)] for a,b in zip(boundary,np.roll(boundary,-1)))
    if not output:
        raise ValueError("The tile has no usable surface after removing collapsed cut faces.")
    output = np.asarray(output,dtype=int)
    _,unique = np.unique(np.sort(output,axis=1),axis=0,return_index=True)
    output = output[np.sort(unique)]
    points = np.asarray(points)
    area = np.linalg.norm(np.cross(points[output[:,1]]-points[output[:,0]],
                                   points[output[:,2]]-points[output[:,0]]),axis=1)
    longest = np.maximum.reduce([np.linalg.norm(points[output[:,a]]-points[output[:,b]],axis=1)
                                 for a,b in ((0,1),(1,2),(2,0))])
    output = output[area > tolerance*np.maximum(longest,tolerance)]
    if not len(output):
        raise ValueError("The tile has no usable surface after removing collapsed cut faces.")
    return _separate_vertex_fans(points,output)


def _tile_shell(tile, center, normal, thickness, curved):
    tile = _conform_tile(tile)
    xyz = np.asarray(tile.points,dtype=float)
    faces = tile.faces.reshape(-1,4)[:,1:].copy()
    vectors = np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
    if vectors.sum(axis=0)@normal < 0:
        faces = faces[:,[0,2,1]]
        vectors = -vectors
    if not curved:
        if np.any(vectors@normal < -1e-10):
            raise ValueError("A tile folds across its tangent plane. Enable Preserve surface curvature or use smaller tiles.")
        xyz = xyz-((xyz-center)@normal)[:,None]*normal
        vectors = np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
        faces = faces[np.linalg.norm(vectors,axis=1)>1e-12]
        directions = np.broadcast_to(normal,xyz.shape)
    else:
        directions = np.zeros_like(xyz)
        for k in range(3):
            np.add.at(directions,faces[:,k],vectors)
        lengths = np.linalg.norm(directions,axis=1)
        directions = np.where((lengths>1e-12)[:,None],directions/np.maximum(lengths[:,None],1e-12),normal)
    if not len(faces):
        raise ValueError("A tile collapses when flattened. Enable Preserve surface curvature.")
    edges = np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,first,counts = np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    if np.any(counts>2):
        raise ValueError("A tile has non-manifold edges. The cut surface needs additional repair before thickening.")
    boundary = edges[first[counts==1]]
    n = len(xyz)
    vertices = np.vstack([xyz-directions*thickness/2,xyz+directions*thickness/2])
    walls = np.vstack([np.c_[boundary[:,0],boundary[:,1],boundary[:,1]+n],
                       np.c_[boundary[:,0],boundary[:,1]+n,boundary[:,0]+n]])
    shell = np.vstack([faces[:,[0,2,1]],faces+n,walls])
    result=pv.PolyData(vertices,np.c_[np.full(len(shell),3),shell].ravel())
    result.field_data["branch_faces_separated"]=tile.field_data["branch_faces_separated"]
    return result


def voronoi_solids(mesh, cloud, surface, thickness=.1, weighted=False,
                   min_thickness=.03, max_thickness=.3, preserve_curvature=False):
    """Closed tile shells, including walls around circulation holes.

    Thickness is total thickness, centred on the tile or its seed tangent plane.
    Sound affects thickness only, using each tile's original generating point.
    """
    _range(min_thickness,max_thickness) if weighted else _range(thickness,thickness)
    if not mesh.n_cells or "voronoi_cell" not in mesh.cell_data or "center_source_point" not in mesh.cell_data:
        raise ValueError("Generate the Voronoi mesh first.")
    if cloud is None or surface is None or not surface.n_cells:
        raise ValueError("The source points and metaball surface are required.")
    reference = surface.triangulate()
    labels = np.asarray(mesh.cell_data["voronoi_cell"])
    unique = np.unique(labels)
    # Find local normals at generating points once for all tiles.
    origins = np.array([int(mesh.cell_data["center_source_point"][np.flatnonzero(labels==label)[0]]) for label in unique])
    if np.any(origins<0) or np.any(origins>=cloud.n_points):
        raise ValueError("Voronoi source points changed. Regenerate the mesh.")
    centers = np.asarray(cloud.points,dtype=float)[origins]
    source_faces = reference.faces.reshape(-1,4)[:,1:][reference.find_closest_cell(centers)]
    corners = np.asarray(reference.points,dtype=float)[source_faces]
    normals = np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
    lengths = np.linalg.norm(normals,axis=1)
    if np.any(lengths<1e-12):
        raise ValueError("The metaball contains a degenerate face at a tile centre.")
    normals /= lengths[:,None]
    energy = _weights(cloud,origins,weighted)
    sizes = min_thickness+(max_thickness-min_thickness)*energy if weighted else np.full(len(unique),thickness)
    vertices, faces, energies, ids, sources, size_values = [], [], [], [], [], []
    offset = 0
    face_count = 0
    for label,source,center,normal,value,size in zip(unique,origins,centers,normals,energy,sizes):
        tile = mesh.extract_cells(np.flatnonzero(labels==label)).extract_surface(algorithm="dataset_surface").triangulate()
        try:
            shell = _tile_shell(tile,center,normal,size,preserve_curvature)
        except ValueError as error:
            raise ValueError(f"Voronoi tile {int(label)}: {error}") from error
        cells = shell.faces.reshape(-1,4)[:,1:]
        vertices.append(shell.points)
        faces.append(cells+offset)
        offset += shell.n_points
        energies.extend([value]*len(cells))
        ids.extend([label]*len(cells))
        sources.extend([source]*len(cells))
        size_values.extend([size]*len(cells))
        face_count += len(cells)
        if face_count>4000000:
            raise ValueError("These solids exceed four million triangles. Use fewer Voronoi cells or points.")
    result = pv.PolyData(np.vstack(vertices),np.c_[np.full(face_count,3),np.vstack(faces)].ravel())
    result.cell_data["voronoi_cell"] = np.asarray(ids)
    result.cell_data["center_source_point"] = np.asarray(sources)
    result.cell_data["center_energy"] = np.asarray(energies)
    result.cell_data["thickness_m"] = np.asarray(size_values)
    result.field_data["preserve_curvature"] = np.array([preserve_curvature],dtype=np.uint8)
    result.field_data["source_method"] = np.array(["voronoi"])
    return result
