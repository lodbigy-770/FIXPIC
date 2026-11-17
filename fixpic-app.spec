# -*- mode: python ; coding: utf-8 -*-
# M5: אריזת הממשק ב-PyInstaller, מצב onedir (ARCHITECTURE §אריזה). בנייה (אחרי יצירת ההודעות):
#   python tools/m5/third_party_notices.py
#   pyinstaller fixpic-app.spec --noconfirm --distpath dist --workpath build/pyi
#   python tools/m5/check_dist.py dist/FixPic
# ספריות Qt נשארות קבצים נפרדים בתיקייה (קישור דינמי, כנדרש ב-LGPL); בלי UPX (חשד אנטי-וירוס).
from pathlib import Path

ROOT = Path(SPECPATH)
notices = ROOT / 'build' / 'THIRD_PARTY_NOTICES.txt'
if not notices.is_file():
    raise SystemExit('חסר build/THIRD_PARTY_NOTICES.txt: הריצו קודם python tools/m5/third_party_notices.py')

a = Analysis(
    [str(ROOT / 'packaging' / 'fixpic_app.py')],
    pathex=[str(ROOT / 'src')],
    binaries=[],
    datas=[
        (str(ROOT / 'models'), 'models'),
        (str(ROOT / 'src' / 'fixpic' / 'strings' / 'he.json'), 'fixpic/strings'),
        (str(ROOT / 'src' / 'fixpic' / 'ui' / 'assets'), 'fixpic/ui/assets'),
        (str(ROOT / 'licenses'), 'licenses'),
        (str(notices), '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # LICENSES-REVIEW.md: בלי טלמטריה של Intel, כלי ההמרה של OpenVINO ורכיבי PySide6 שאינם בשימוש.
    excludes=['openvino_telemetry', 'openvino.tools', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtQuick', 'PySide6.QtQml',
              'PySide6.Qt3DCore', 'PySide6.QtMultimedia', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FixPic',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / 'src' / 'fixpic' / 'ui' / 'assets' / 'fixpic.ico'),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='FixPic')
