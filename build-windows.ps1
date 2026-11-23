param([string]$Python)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$environmentPath = Join-Path $PSScriptRoot '.build\venv'
$buildPython = Join-Path $environmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $buildPython)) {
    if ($Python) {
        & $Python -m venv $environmentPath
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        uv --cache-dir (Join-Path $PSScriptRoot '.build\uv-cache') venv --python 3.12 --seed $environmentPath
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv $environmentPath
    } else {
        throw 'Python 3.12 is needed on the build computer. Install it, or use -Python with its full executable path.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating the build environment failed.' }
}
& $buildPython -m pip install -r (Join-Path $PSScriptRoot 'packaging\requirements-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing build dependencies failed.' }
& $buildPython (Join-Path $PSScriptRoot 'packaging\build_windows.py')
if ($LASTEXITCODE -ne 0) { throw 'Packaging or the launch check failed. The build is not ready to share.' }

