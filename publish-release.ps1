$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.build\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'The packaging Python environment is missing. Run build-windows.ps1 first.' }
& $python (Join-Path $PSScriptRoot 'packaging\publish_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Publishing stopped. Read the message above; no force push was used.' }

