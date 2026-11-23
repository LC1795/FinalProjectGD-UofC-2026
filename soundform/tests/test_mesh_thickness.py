import numpy as np
import pyvista as pv
import pytest
from soundform.geometry.mesh_thickness import delaunay_frame, voronoi_solids, _tile_shell
from soundform.geometry.voronoi_filters import subtract_circulation


def test_delaunay_unique_edges_nodes_and_tapered_radii():
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
    cloud=pv.PolyData(xyz)
    cloud["energy"]=np.array([0,.5,1,.25])
    mesh=pv.PolyData(xyz,np.array([3,0,1,2,3,1,3,2]))
    mesh["source_point"]=np.arange(4)
    uniform=delaunay_frame(mesh,cloud,radius=.05)
    assert np.allclose(uniform.field_data["node_radius_m"],.05)
    assert uniform.field_data["edge_source_points"].size==10 # Five unique shared edges.
    assert uniform.n_open_edges==0
    weighted=delaunay_frame(mesh,cloud,weighted=True,min_radius=.1,max_radius=.3)
    assert np.allclose(weighted.field_data["node_radius_m"],[.1,.2,.3,.15])
    assert np.allclose(np.linalg.norm(weighted.points[:12]-xyz[0],axis=1),.1)
    assert np.allclose(np.linalg.norm(weighted.points[12:24]-xyz[1],axis=1),.2)
    assert weighted.n_open_edges==0
    with pytest.raises(ValueError,match="minimum"):
        delaunay_frame(mesh,cloud,weighted=True,min_radius=.3,max_radius=.1)


def test_flat_and_curved_tiles_form_closed_shells():
    sphere=pv.Sphere(theta_resolution=16,phi_resolution=12)
    ids=np.flatnonzero(sphere.cell_centers().points[:,2]>.3)
    tile=sphere.extract_cells(ids).extract_surface(algorithm="dataset_surface").triangulate()
    normal=np.array([0.,0,1])
    flat=_tile_shell(tile,np.array([0,0,.5]),normal,.1,False)
    curved=_tile_shell(tile,np.array([0,0,.5]),normal,.1,True)
    for result in (flat,curved):
        assert result.n_open_edges==0
        assert np.isfinite(result.points).all()
    assert np.ptp(flat.points[:flat.n_points//2,2])<1e-9
    assert np.ptp(curved.points[:curved.n_points//2,2])>.05
    half=curved.n_points//2
    assert np.allclose(np.linalg.norm(curved.points[half:]-curved.points[:half],axis=1),.1)


def test_voronoi_center_weight_sets_constant_tile_thickness_and_color():
    surface=pv.Plane(i_size=2,j_size=2,i_resolution=2,j_resolution=2).triangulate()
    centers=surface.cell_centers().points
    labels=(centers[:,0]>0).astype(int)
    cloud=pv.PolyData(np.array([[-.5,0,0],[.5,0,0]]))
    cloud["energy"]=np.array([0.,1.])
    surface.cell_data["voronoi_cell"]=labels
    surface.cell_data["center_source_point"]=labels
    result=voronoi_solids(surface,cloud,surface,weighted=True,min_thickness=.1,max_thickness=.3)
    assert result.n_open_edges==0
    for label,thickness in ((0,.1),(1,.3)):
        tile=result.extract_cells(np.flatnonzero(result["voronoi_cell"]==label))
        assert np.ptp(tile.points[:,2])==pytest.approx(thickness)
        assert np.allclose(tile.cell_data["thickness_m"],thickness)
        assert np.allclose(tile.cell_data["center_energy"],label)


def test_circulation_hole_is_not_capped_by_solid_tile():
    plane=pv.Plane(i_size=4,j_size=4,i_resolution=3,j_resolution=3).triangulate()
    plane.cell_data["voronoi_cell"]=np.zeros(plane.n_cells,int)
    plane.cell_data["center_source_point"]=np.zeros(plane.n_cells,int)
    curve=(np.array([[0.,0,-1],[0,0,1]]),np.array([2,0,1]),.3)
    cut=subtract_circulation(plane,[curve])
    cloud=pv.PolyData(np.array([[-1.,0,0]]))
    cloud["energy"]=np.array([.7])
    result=voronoi_solids(cut,cloud,plane,thickness=.1)
    assert result.n_open_edges==0
    _,closest=result.find_closest_cell([0,0,0],return_closest_point=True)
    assert np.linalg.norm(closest)>=.3-1e-8


@pytest.mark.parametrize("curved", [False, True])
def test_cut_vertex_precision_and_duplicate_faces_are_repaired(curved):
    # Float32 cut coordinates can differ below their meaningful precision.
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0],
                  [1,0,3e-8],[.5,0,0]],dtype=np.float32)
    faces=np.array([[0,1,2],[4,3,2],[0,1,2],[1,4,5]])
    tile=pv.PolyData(xyz,np.c_[np.full(len(faces),3),faces].ravel())
    result=_tile_shell(tile,np.zeros(3),np.array([0.,0,1]),.1,curved)
    assert result.n_open_edges==0
    assert result.volume==pytest.approx(.1,rel=1e-5)


def test_touching_surface_sheets_are_closed_independently():
    xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,0],[.5,.5,.1]])
    faces=np.array([[0,1,2],[1,0,3],[0,1,4]])
    tile=pv.PolyData(xyz,np.c_[np.full(3,3),faces].ravel())
    result=_tile_shell(tile,np.zeros(3),np.array([0.,0,1]),.1,True)
    assert result.n_open_edges==0
    assert result.field_data["branch_faces_separated"][0]==3
    assert np.isfinite(result.points).all()
