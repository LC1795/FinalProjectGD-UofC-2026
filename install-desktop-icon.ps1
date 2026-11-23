param([string]$LogoPath)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = 'C:\workspace\FinalProjectGD-UofC-2026-main\.venv\Scripts\python.exe'
if (-not $LogoPath) {
    $candidates = @(
        (Join-Path $projectRoot 'soundform\assets\soundform-logo.png'),
        (Join-Path $env:USERPROFILE 'Downloads\Coarse Voronoi Sphere on Black.png')
    )
    $LogoPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $LogoPath -or -not (Test-Path -LiteralPath $LogoPath)) {
    Add-Type -AssemblyName System.Windows.Forms
    $picker = New-Object System.Windows.Forms.OpenFileDialog
    $picker.Title = 'Select the approved Soundform logo image'
    $picker.Filter = 'Image files|*.png;*.jpg;*.jpeg;*.webp|All files|*.*'
    $picker.InitialDirectory = Join-Path $env:USERPROFILE 'Downloads'
    try {
        if ($picker.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
            Write-Host 'Installation cancelled.'
            return
        }
        $LogoPath = $picker.FileName
    } finally {
        $picker.Dispose()
    }
}
$assetDirectory = Join-Path $projectRoot 'soundform\assets'
New-Item -ItemType Directory -Force -Path $assetDirectory | Out-Null
@'
import sys
from pathlib import Path
from PIL import Image
source, destination = Path(sys.argv[1]), Path(sys.argv[2])
with Image.open(source) as image:
    image = image.convert('RGBA')
    if image.width != image.height:
        raise ValueError('The logo must be square.')
    image.save(destination / 'soundform-logo.png')
    image.save(destination / 'soundform.ico', format='ICO',
               sizes=[(n,n) for n in (16,24,32,48,64,128,256)])
'@ | & $python - $LogoPath $assetDirectory
if ($LASTEXITCODE -ne 0) { throw 'Icon conversion failed.' }
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $desktop 'Soundform.lnk'))
$shortcut.TargetPath = Join-Path (Split-Path $python) 'pythonw.exe'
$shortcut.Arguments = '"' + (Join-Path $projectRoot 'launch_soundform.pyw') + '"'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = (Join-Path $assetDirectory 'soundform.ico') + ',0'
$shortcut.Description = 'Soundform — sound-driven architectural geometry'
$shortcut.Save()
Write-Host 'Soundform logo and desktop shortcut installed. Restart Soundform to see its taskbar icon.'

