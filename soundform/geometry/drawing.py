"""Point, polyline and clamped control-point spline geometry."""
import numpy as np
import pyvista as pv
from scipy.interpolate import BSpline


def drawing_geometry(kind,points):
    controls=np.asarray(points,dtype=float)
    minimum={"point":1,"line":2,"spline":3}[kind]
    if controls.ndim!=2 or controls.shape[1]!=3 or len(controls)<minimum or not np.isfinite(controls).all():
        raise ValueError(f"{kind.title()} requires at least {minimum} finite points.")
    if kind=="point":
        return pv.PolyData(controls)
    if kind=="line":
        xyz=controls
    else:
        degree=min(3,len(controls)-1)
        interior=np.linspace(0,1,len(controls)-degree+1)[1:-1]
        knots=np.r_[np.zeros(degree+1),interior,np.ones(degree+1)]
        xyz=BSpline(knots,controls,degree)(np.linspace(0,1,max(32,min(2048,16*(len(controls)-1)))))
    mesh=pv.PolyData(xyz)
    mesh.verts=np.empty(0,dtype=np.int64)
    mesh.lines=np.r_[len(xyz),np.arange(len(xyz))].astype(np.int64)
    mesh.field_data["curve_type"]=np.array([kind])
    mesh.field_data["control_points"]=controls
    return mesh


def intersect_plane(start,end,origin,normal):
    ray=np.asarray(end)-start
    denominator=float(ray@normal)
    if abs(denominator)<1e-12:
        return None
    distance=float((np.asarray(origin)-start)@normal)/denominator
    if distance<0:
        return None
    return np.asarray(start)+distance*ray
