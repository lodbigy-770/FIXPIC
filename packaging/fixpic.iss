; M5: מתקין Inno Setup יחיד שעוטף את dist\FixPic (פלט PyInstaller onedir, כולל המודל) בקובץ EXE אחד.
; משתמש הקצה מוריד ומריץ קובץ אחד; אין צורך להכניס את המודל ידנית.
;
; בנייה (בסדר הזה, ב-Windows; ראו גם DEVELOPMENT_LOG.md):
;   1. python tools\m5\third_party_notices.py
;   2. pyinstaller fixpic-app.spec --noconfirm --distpath dist --workpath build\pyi
;   3. python tools\m5\check_dist.py dist\FixPic
;   4. iscc /DMyAppVersion=0.1.0 packaging\fixpic.iss
; פלט: dist\installer\FixPicSetup-0.1.0.exe
;
; התקנה בלי הרשאות מנהל ל-%LOCALAPPDATA%\Programs\FixPic (כמו תיקיית הלוגים הקיימת
; ב-fixpic/diagnostics.py, %LOCALAPPDATA%\FIXPIC\logs) — כלי אישי ולא מסחרי, בלי צורך ב-UAC.
; ספריות Qt נשארות קבצים נפרדים בתוך {app} (קישור דינמי, כנדרש ב-LGPL; ראו check_dist.py).

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#define MyAppName "FixPic"
#define MyAppId "{5A2ACA77-157D-4C60-9FA2-AE82BC02E680}"
#define MyAppExeName "FixPic.exe"
#define SourceDist "..\dist\FixPic"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppName}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=FixPicSetup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesInstallIn64BitMode=x64compatible
; המודל (כ-200MB) כבר דחוס; דחיסה כפולה לא עוזרת אך לא פוגעת, והושארה לאחידות עם שאר הקבצים.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "צור קיצור בשולחן העבודה"; GroupDescription: "קיצורים נוספים:"

[Files]
Source: "{#SourceDist}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\הסר התקנה של {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "הפעל את {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; יומנים/קבצי קריסה ב-%LOCALAPPDATA%\FIXPIC\logs (diagnostics.py) נשארים במכוון להסרה —
; המשתמש עשוי לרצות לשמור אותם אחרי הסרת האפליקציה.
