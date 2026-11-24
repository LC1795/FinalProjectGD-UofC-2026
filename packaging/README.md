# SoundForm V2.0 for Windows

Download **SoundForm-V2.0-Setup-x64.exe** from GitHub Releases and run it.
The same installer is included in **SoundForm-V2.0-Windows-x64.zip**, along
with installation instructions. Extract that ZIP before running Setup.

Setup installs for the current user and adds a Start menu shortcut. It also
offers a desktop shortcut. Windows Settings can uninstall the app. The package
includes Python and the application dependencies. It supports 64-bit Windows
10 and 11. The app and installer are unsigned.

## Build and publish

On 64-bit Windows with Python 3.12 and Inno Setup 6.3 or later, run:

```powershell
.\build-windows.ps1
```

Set `$env:ISCC` to the full path to `ISCC.exe` if it lives outside the default
installation locations. The build checks the bundled app before compiling Setup.
The installer, ZIP, checksum, and launch report are written to `dist`.

The **Build Windows app** workflow can also run on GitHub Actions. It tests the
source, builds Setup, and checks installation, packaged startup, and uninstall
before uploading the resulting files as an Actions artifact.

Run `.\publish-release.ps1` from a clean main checkout after building to
publish the installer, ZIP, and checksum to GitHub Releases. The automatic
GitHub source archives contain the source tree.
