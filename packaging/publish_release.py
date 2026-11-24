"""Push main and upload the tested ZIP without requiring GitHub CLI."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "LC1795/FinalProjectGD-UofC-2026"
TAG = "soundform-v2.0.0"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def local_artifacts():
    archive = ROOT / "dist" / "SoundForm-V2.0-Windows-x64.zip"
    checksum = archive.with_suffix(".zip.sha256")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum.read_text().split()[0] != digest:
        raise RuntimeError("The ZIP does not match its checksum. Rebuild before publishing.")
    installer = archive.with_name("SoundForm-V2.0-Setup-x64.exe")
    with zipfile.ZipFile(archive) as zipped, installer.open("rb") as standalone:
        build = json.loads(zipped.read("build-info.json"))
        revision = build["revision"]
        with zipped.open(installer.name) as packaged:
            if hashlib.file_digest(packaged, "sha256").digest() != hashlib.file_digest(standalone, "sha256").digest():
                raise RuntimeError("The standalone installer differs from the ZIP. Rebuild before publishing.")
    if build.get("dirty") is not False or revision != git("rev-parse", "HEAD"):
        raise RuntimeError("Rebuild the installer from the current clean commit before publishing.")
    if git("branch", "--show-current") != "main" or git("status", "--porcelain"):
        raise RuntimeError("Publish from a clean main branch. Commit or save local edits first.")
    return archive, checksum, installer, revision


def credentials():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # Reuse the credential helper that authenticates ordinary HTTPS git pushes.
    # Credential output is captured privately and is never printed or saved.
    result = subprocess.run(["git", "credential", "fill"], cwd=ROOT,
        input=f"protocol=https\nhost=github.com\npath={REPO}.git\n\n",
        text=True, capture_output=True)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise RuntimeError("GitHub credentials unavailable. Sign in through Git Credential Manager, then retry.")
    return fields["password"]


def request(token, method, path, payload=None, upload=None):
    host = "uploads.github.com" if upload else "api.github.com"
    connection = http.client.HTTPSConnection(host, timeout=600)
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "Soundform-release",
               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        if upload:
            headers.update({"Content-Type": "application/octet-stream",
                            "Content-Length": str(upload.stat().st_size)})
            connection.putrequest(method, path)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            with upload.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    connection.send(chunk)
        else:
            headers["Content-Type"] = "application/json"
            connection.request(method, path, body=json.dumps(payload).encode() if payload is not None else None,
                               headers=headers)
        response = connection.getresponse()
        body = response.read()
        result = json.loads(body) if body else {}
        if response.status >= 400:
            raise RuntimeError(f"GitHub HTTP {response.status}: {str(result.get('message', 'Request failed')).replace(token, '[redacted]')}")
        return result
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    archive, checksum, installer, revision = local_artifacts()
    print(f"Verified ZIP: {archive.name} ({archive.stat().st_size:,} bytes), source {revision[:7]}", flush=True)
    if args.check_only:
        return
    subprocess.run(["git", "fetch", "origin"], cwd=ROOT, check=True)
    if git("remote", "get-url", "origin").rstrip("/").removesuffix(".git") != f"https://github.com/{REPO}":
        raise RuntimeError("Unexpected origin repository; stopped.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=ROOT).returncode:
        raise RuntimeError("GitHub main has new changes. Merge them before retrying; no files have been overwritten.")
    token = credentials()
    base = f"/repos/{REPO}"
    # Check any existing tag before creating a draft or publishing anything.
    refs = git("ls-remote", "--tags", "origin", f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}")
    if refs:
        tagged = [line.split()[0] for line in refs.splitlines()]
        if tagged[-1] != revision:
            raise RuntimeError(f"{TAG} already points to a different revision; stopped.")
    releases = request(token, "GET", base + "/releases?per_page=100")
    release = next((item for item in releases if item["tag_name"] == TAG), None)
    if release and not release["draft"]:
        raise RuntimeError(f"This version is already published: {release['html_url']}. Existing releases are left unchanged.")
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
    if release is None:
        release = request(token, "POST", base + "/releases", {
            "tag_name": TAG, "target_commitish": revision, "name": "SoundForm V2.0 — Windows",
            "draft": True, "prerelease": False,
            "body": (ROOT / "packaging" / "release-notes.md").read_text(encoding="utf-8")})
    elif release.get("target_commitish") != revision:
        raise RuntimeError("The existing draft targets another revision; stopped.")
    assets = request(token, "GET", base + f"/releases/{release['id']}/assets?per_page=100")
    for path in (installer, archive, checksum):
        with path.open("rb") as stream:
            digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
        existing = next((item for item in assets if item["name"] == path.name), None)
        if existing:
            if existing.get("digest") != digest or existing["size"] != path.stat().st_size:
                raise RuntimeError(f"Draft asset {path.name} differs. Left unchanged; resolve it before retrying.")
            continue
        print(f"Uploading {path.name}...", flush=True)
        asset = request(token, "POST", base + f"/releases/{release['id']}/assets?name={quote(path.name)}", upload=path)
        if asset["size"] != path.stat().st_size or asset.get("digest", digest) != digest:
            raise RuntimeError("Uploaded asset verification failed; release remains a draft.")
    published = request(token, "PATCH", base + f"/releases/{release['id']}", {"draft": False})
    print("Published: " + published["html_url"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Publishing stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
