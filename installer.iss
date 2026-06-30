[Setup]
AppName=Brushout
AppVersion=1.0.0
AppPublisher=Shane Turner
DefaultDirName={userappdata}\Brushout
DefaultGroupName=Brushout
OutputDir=Output
OutputBaseFilename=Brushout-Setup
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
MinVersion=10.0

[Files]
Source: "dist\Brushout\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Brushout"; Filename: "{app}\Brushout.exe"
Name: "{commondesktop}\Brushout"; Filename: "{app}\Brushout.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Run]
Filename: "{app}\Brushout.exe"; Description: "{cm:LaunchProgram,Brushout}"; Flags: nowait postinstall skipifsilent
