"""Bundle, check and zip the app. Invoke with the prepared build Python."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Build the Windows application on Windows.")
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--distpath", str(output), "--workpath", str(root / ".build" / "pyinstaller"),
                    str(root / "packaging" / "Soundform.spec")], cwd=root, check=True)
    bundle = output / "Soundform"
    shutil.copy2(root / "packaging" / "START-HERE.txt", bundle)
    (bundle / "build-info.json").write_text(json.dumps({"revision": revision,
        "python": sys.version, "platform": sys.platform}, indent=2))
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
    archive = output / "Soundform-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(output))
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Ready to share: {archive}")


if __name__ == "__main__":
    main()

