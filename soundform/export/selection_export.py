"""Export selected meshes, polylines and point clouds in local site metres."""
from pathlib import Path
import numpy as np
from soundform.errors import ExportError


def _lines(packed):
    offset = 0
    while offset < len(packed):
        count = int(packed[offset])
        yield packed[offset+1:offset+1+count]
        offset += count+1


def export_selection(datasets, path):
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in (".3dm", ".obj", ".stl"):
        raise ExportError("Choose .3dm, .obj, or .stl.")
    if not datasets:
        raise ExportError("Nothing is selected.")
    for _, mesh in datasets:
        if not mesh.n_points or not np.isfinite(mesh.points).all():
            raise ExportError("The selection contains empty or invalid geometry.")
    if suffix == ".stl" and any(m.n_lines or not len(m.faces) for _,m in datasets):
        raise ExportError("STL supports surfaces only. Use Rhino or OBJ for curves and points.")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if suffix == ".3dm":
        import rhino3dm as rhino
        document = rhino.File3dm()
        document.Settings.ModelUnitSystem = rhino.UnitSystem.Meters
        for name, data in datasets:
            attr = rhino.ObjectAttributes()
            attr.Name = str(name)
            if len(data.faces):
                triangles = data.triangulate(pass_lines=True,pass_verts=True)
                mesh = rhino.Mesh()
                for xyz in triangles.points:
                    mesh.Vertices.Add(*map(float,xyz))
                for face in triangles.faces.reshape(-1,4)[:,1:]:
                    mesh.Faces.AddFace(*map(int,face))
                # Store the displayed center-energy colours as mesh vertex colours.
                if "center_energy" in triangles.cell_data:
                    from matplotlib import colormaps
                    # Unweld per triangle so neighbouring tiles keep distinct colours.
                    coloured = rhino.Mesh()
                    for face, energy in zip(triangles.faces.reshape(-1,4)[:,1:],triangles.cell_data["center_energy"]):
                        rgba = tuple(int(round(c*255)) for c in colormaps["magma"](float(np.clip(energy,0,1))))
                        start = len(coloured.Vertices)
                        for vertex in triangles.points[face]:
                            coloured.Vertices.Add(*map(float,vertex))
                            coloured.VertexColors.Add(*rgba[:3])
                        coloured.Faces.AddFace(start,start+1,start+2)
                    mesh = coloured
                document.Objects.AddMesh(mesh,attr)
            spline = "curve_type" in data.field_data and str(data.field_data["curve_type"][0])=="spline" and "control_points" in data.field_data
            if spline and data.n_lines:
                controls=[rhino.Point3d(*map(float,p)) for p in data.field_data["control_points"]]
                document.Objects.AddCurve(rhino.NurbsCurve.Create(False,min(3,len(controls)-1),controls),attr)
            else:
                for ids in _lines(data.lines):
                    document.Objects.AddCurve(rhino.PolylineCurve([rhino.Point3d(*map(float,p)) for p in data.points[ids]]),attr)
            if not len(data.faces) and not data.n_lines:
                if data.n_points == 1:
                    document.Objects.AddPoint(rhino.Point3d(*map(float,data.points[0])),attr)
                else:
                    cloud = rhino.PointCloud()
                    for point in data.points:
                        cloud.Add(rhino.Point3d(*map(float,point)))
                    document.Objects.AddPointCloud(cloud,attr)
        if not document.Write(str(destination),8):
            raise ExportError("Could not write the Rhino selection.")
    elif suffix == ".obj":
        rows, offset = ["# Soundform selection; coordinates in metres"], 1
        for name,data in datasets:
            rows.append("o "+str(name).replace("\n"," ").replace("\r"," "))
            rows.extend("v "+" ".join(format(float(v),".17g") for v in xyz) for xyz in data.points)
            for face in data.triangulate().faces.reshape(-1,4)[:,1:]:
                rows.append("f "+" ".join(str(int(i)+offset) for i in face))
            for ids in _lines(data.lines):
                rows.append("l "+" ".join(str(int(i)+offset) for i in ids))
            if not len(data.faces) and not data.n_lines:
                rows.extend("p "+str(i+offset) for i in range(data.n_points))
            offset += data.n_points
        destination.write_text("\n".join(rows)+"\n",encoding="utf-8")
    else:
        import trimesh
        from soundform.export.mesh_export import export_mesh
        meshes = []
        for _,data in datasets:
            tri = data.triangulate()
            meshes.append(trimesh.Trimesh(tri.points,tri.faces.reshape(-1,4)[:,1:],process=False))
        export_mesh(trimesh.util.concatenate(meshes),str(destination))
    if not destination.is_file() or not destination.stat().st_size:
        raise ExportError("The exported file is missing or empty.")
    return dict(path=str(destination),objects=len(datasets))
