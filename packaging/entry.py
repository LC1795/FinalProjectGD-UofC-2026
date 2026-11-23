"""Frozen app entry point with startup diagnostics and an automated launch check."""
import json
import os
from pathlib import Path
import sys
import traceback


def smoke_test(report_path):
    # This mode never contacts an AI service or requires an audio device.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import tempfile
    import numpy as np
    import pyvista as pv
    import rhino3dm
    import sounddevice
    import soundfile
    from PySide6.QtWidgets import QApplication
    from soundform.branding import configure_branding
    from soundform.geometry.mesh_thickness import delaunay_frame, _tile_shell
    from soundform.ui.main_window import MainWindow

    app = QApplication([])
    configure_branding(app)
    assert not app.windowIcon().isNull(), "Bundled logo missing"
    window = MainWindow()
    app.processEvents()
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    cloud = pv.PolyData(xyz)
    cloud["energy"] = np.array([0., .5, 1.])
    triangle = pv.PolyData(xyz, [3, 0, 1, 2])
    triangle["source_point"] = np.arange(3)
    assert delaunay_frame(triangle, cloud).n_open_edges == 0
    assert _tile_shell(triangle, xyz.mean(axis=0), np.array([0., 0, 1]), .1, False).n_open_edges == 0
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "test.wav"
        soundfile.write(wav, np.zeros((800, 4)), 8000)
        assert soundfile.read(wav)[0].shape == (800, 4)
        model = rhino3dm.File3dm()
        model.Objects.AddPoint(0, 0, 0)
        path = Path(directory) / "test.3dm"
        assert model.Write(str(path), 8)
        assert len(rhino3dm.File3dm.Read(str(path)).Objects) == 1
    window.close()
    app.processEvents()
    Path(report_path).write_text(json.dumps({"ok": True, "checks": [
        "Qt window", "logo", "VTK", "Delaunay struts", "Voronoi shell",
        "WAV read/write", "Rhino read/write", "audio backend import"]}, indent=2))
    return 0


def run():
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
            return smoke_test(sys.argv[2])
        from soundform.main import main
        return main()
    except Exception:
        detail = traceback.format_exc()
        if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
            Path(sys.argv[2]).write_text(json.dumps({"ok": False, "error": detail}, indent=2))
            return 1
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Soundform"
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "startup-error.log"
        log.write_text(detail, encoding="utf-8")
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"Soundform could not start. Details saved to:\n{log}",
                                       "Soundform", 0x10)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())

