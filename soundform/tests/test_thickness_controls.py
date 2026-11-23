import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtTest import QTest
from soundform.ui.main_window import MainWindow
from soundform.export.selection_export import export_selection


def test_thickness_actions_finish_in_layers_and_export_and_invalidate(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    errors=[]
    monkeypatch.setattr(QMessageBox,"warning",lambda *args:errors.append(args[-1]))
    w=MainWindow()
    def finish():
        for _ in range(500):
            QTest.qWait(20)
            if not getattr(w,"_busy",False) and not w._worker.isRunning():
                break
        assert not getattr(w,"_busy",False)
        assert not errors,errors
    try:
        w.show()
        xyz=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        points=pv.PolyData(xyz);points["energy"]=np.array([0,.3,.7,1])
        base=pv.PolyData(xyz,np.array([3,0,1,2,3,1,3,2]))
        base["source_point"]=np.arange(4)
        w.viewport._cloud_points=points
        w.viewport._cloud_surface=base.copy()
        assert not w._thickness_groups["delaunay"].isEnabled()
        w.viewport.show_reconstruction("delaunay",base)
        w._rebuild_layers()
        assert w._thickness_groups["delaunay"].isEnabled()
        w._thicken_mesh("delaunay",True)
        finish()
        frame=w.viewport._reconstructions["delaunay-frame"]
        assert frame.n_open_edges==0
        export_selection([("Frame",frame)],tmp_path/"frame.stl")
        assert any(n.text(0)=="Delaunay struts and spheres" for n in w._export_rows())
        v=base.copy()
        v.cell_data["voronoi_cell"]=np.zeros(v.n_cells,int)
        v.cell_data["center_source_point"]=np.zeros(v.n_cells,int)
        v.cell_data["center_energy"]=np.zeros(v.n_cells)
        w.viewport.show_reconstruction("voronoi",v,color_by_cell=True)
        w._rebuild_layers()
        w._thickness_inputs["voronoi"][3].setChecked(True)
        w._thicken_mesh("voronoi",True)
        finish()
        solid=w.viewport._reconstructions["voronoi-solid"]
        assert solid.n_open_edges==0
        assert solid.field_data["preserve_curvature"][0]==1
        export_selection([("Tiles",solid)],tmp_path/"tiles.3dm")
        assert any(n.text(0)=="Voronoi solid tiles" for n in w._export_rows())
        w.viewport.show_reconstruction("voronoi",v,color_by_cell=True)
        w._rebuild_layers()
        assert "voronoi-solid" not in w.viewport._reconstructions
        assert not any(n.text(0)=="Voronoi solid tiles" for n in w._export_rows())
        w.viewport.remove_reconstruction("delaunay")
        w._rebuild_layers()
        assert "delaunay-frame" not in w.viewport._reconstructions
        assert not w._thickness_groups["delaunay"].isEnabled()
    finally:
        w.close()
