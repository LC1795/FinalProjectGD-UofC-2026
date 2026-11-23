"""Windows folder bundle: assets and dynamically loaded geometry modules included."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
datas = [(str(root / "soundform" / "assets"), "soundform/assets")]
binaries = []
hiddenimports = ["PySide6.QtOpenGLWidgets", "matplotlib.backends.backend_qtagg"]
# SciPy's array backend resolves these modules dynamically. Older freezer
# hooks do not discover the vendored FFT/linalg modules in newer SciPy wheels.
hiddenimports += collect_submodules("scipy._external")
hiddenimports += collect_submodules("scipy._lib.array_api_compat")
for package in ("vtkmodules", "pyvista", "pyvistaqt", "rhino3dm"):
    data, binary, hidden = collect_all(package, include_py_files=False,
                                     filter_submodules=lambda name: ".tests" not in name)
    datas += data
    binaries += binary
    hiddenimports += hidden
hiddenimports += collect_submodules("trimesh", filter=lambda name: ".tests" not in name)
for distribution in ("numpy", "scipy", "soundfile", "matplotlib", "PySide6",
                     "pyvista", "pyvistaqt", "trimesh", "rtree", "sounddevice", "rhino3dm"):
    datas += copy_metadata(distribution)

a = Analysis([str(root / "packaging" / "entry.py")], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "pytest", "IPython", "jupyter"],
             hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Soundform",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, icon=str(root / "soundform" / "assets" / "soundform.ico"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Soundform")

