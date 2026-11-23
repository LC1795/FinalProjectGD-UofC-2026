# Windows distribution

Build on 64-bit Windows with Python 3.12 and internet access. The build computer
needs Python; recipients do not. PyInstaller's folder bundle keeps the large Qt
and VTK libraries beside the application rather than extracting them on every
launch. See the [PyInstaller specification documentation](https://pyinstaller.org/en/v6.17.0/spec-files.html).

From PowerShell, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\workspace\Soundform\build-windows.ps1"
```

The script creates an isolated `.build/venv` using uv or the Python launcher,
installs dependencies, builds the executable with the approved logo, runs a
packaged startup check, and writes `dist/Soundform-Windows-x64.zip` plus its SHA-256.
No desktop shortcut on the build computer is changed. No API keys, user settings,
recordings, site models, Python environments or source tests are copied as assets.
Package dependencies and their distribution metadata are included.

If using an installed Python without uv or the Python launcher, pass its path:

```powershell
.\build-windows.ps1 -Python 'C:\path\to\python.exe'
```

Builds use the dependency ranges in `requirements-build.txt`. Each artifact records
the exact resolved dependency versions in `dependencies.txt`, the source commit
in `build-info.json`, and the packaged launch result beside the ZIP. Keep those
files when reporting a build problem. Dependencies are not yet locked, so separate
builds can resolve different versions.

Classmates should extract the whole ZIP, then run `Soundform/Soundform.exe`.
Keep `_internal` beside the EXE. `START-HERE.txt` includes recipient instructions.
The app is unsigned; Windows may show an unknown-publisher warning.

## Build on GitHub

After pushing these files, open **Actions → Build Windows app → Run workflow**
on main. A successful run provides a **Soundform-Windows-x64** downloadable
artifact containing the portable ZIP, checksum and smoke-test report. People with
repository access can download it. Nothing is published as a GitHub Release
automatically. Tags beginning with `soundform-v` also trigger a build.

The packaged check covers Qt creation, bundled icon, VTK geometry, Delaunay struts,
Voronoi extrusion, WAV/Rhino round trips and audio-backend loading. It uses Qt's
offscreen mode, so it does not validate GPU rendering, physical audio output,
interactive workflows, normal application shutdown or AI credentials. Before sharing a release, open the EXE
on a second Windows computer, import a site and recording, generate both meshes,
thicken them, and export a result.
