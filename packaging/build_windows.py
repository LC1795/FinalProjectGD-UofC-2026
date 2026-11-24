"""Bundle, check and zip the app. Invoke with the prepared build Python."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist")
    parser.add_argument("--iscc", default=os.environ.get("ISCC"))
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Build the Windows application on Windows.")
    compiler = args.iscc or shutil.which("ISCC.exe")
    if not compiler:
        for variable in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
            if location := os.environ.get(variable):
                base = Path(location)
                if variable == "LOCALAPPDATA":
                    base /= "Programs"
                candidate = base / "Inno Setup 6" / "ISCC.exe"
                if candidate.is_file():
                    compiler = str(candidate)
                    break
    if not compiler or not Path(compiler).is_file():
        parser.error("Install Inno Setup 6.3 or later, or specify the full path with --iscc.")
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    # A desktop shell can have other apps' DLL directories on PATH. In
    # particular, Poppler's ICU DLL is incompatible with Qt's system ICU.
    # Let PyInstaller's package hooks supply dependency directories instead.
    build_environment = os.environ.copy()
    windows = Path(os.environ["SystemRoot"])
    build_environment["PATH"] = os.pathsep.join(map(str, [Path(sys.executable).parent,
        Path(sys.base_prefix), windows / "System32", windows]))
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--distpath", str(output), "--workpath", str(root / ".build" / "pyinstaller"),
                    str(root / "packaging" / "Soundform.spec")], cwd=root,
                   env=build_environment, check=True)
    bundle = output / "Soundform"
    (bundle / "build-info.json").write_text(json.dumps({"revision": revision,
        "dirty": dirty, "python": sys.version, "platform": sys.platform}, indent=2))
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (bundle / "dependencies.txt").write_text(packages)
    report = output / "smoke-test.json"
    # A stale success report must never validate a failed launch.
    report.write_text('{}')
    launched = subprocess.run([str(bundle / "Soundform.exe"), "--smoke-test", str(report)],
                              cwd=output, timeout=180)
    if launched.returncode:
        raise RuntimeError(f"Packaged launch failed (exit {launched.returncode}): {report.read_text()}")
    if json.loads(report.read_text()).get("ok") is not True:
        raise RuntimeError(f"Packaged launch check failed: {report.read_text()}")
    subprocess.run([compiler, f"/DBundleDir={bundle}", f"/O{output}",
                    str(root / "packaging" / "SoundForm.iss")], check=True)
    installer = output / "SoundForm-V2.0-Setup-x64.exe"
    archive = output / "SoundForm-V2.0-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(installer, installer.name)
        zipped.write(root / "packaging" / "START-HERE.txt", "START-HERE.txt")
        zipped.write(bundle / "build-info.json", "build-info.json")
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Ready to share: {archive}")


if __name__ == "__main__":
    main()

