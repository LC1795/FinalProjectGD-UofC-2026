[Setup]
AppId=LC1795.SoundForm
AppName=SoundForm V2.0
AppVersion=2.0.0
AppPublisher=SoundForm
DefaultDirName={localappdata}\Programs\SoundForm
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile=..\soundform\assets\soundform.ico
UninstallDisplayIcon={app}\Soundform.exe
OutputBaseFilename=SoundForm-V2.0-Setup-x64
Compression=lzma2

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Icons]
Name: "{userprograms}\SoundForm V2.0"; Filename: "{app}\Soundform.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\SoundForm V2.0"; Filename: "{app}\Soundform.exe"; WorkingDir: "{app}"; Tasks: desktopicon
