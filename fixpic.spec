# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/fixpic/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # M5 (LICENSES-REVIEW.md): לא לארוז טלמטריה של Intel וכלי המרת מודלים; בלי רכיבי PySide6 שאינם בשימוש (WebEngine ועוד).
    excludes=['openvino_telemetry', 'openvino.tools', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
              'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.Qt3DCore', 'PySide6.QtMultimedia'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='fixpic',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
