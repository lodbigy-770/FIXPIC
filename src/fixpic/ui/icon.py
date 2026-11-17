"""אייקון היישום: טעינה מ-`assets/` (גם בהרצה מארוזה, תחת `sys._MEIPASS`) והגדרת מזהה יישום ב-Windows.

האייקון נוצר ב-`tools/m5/make_icon.py`. כשל בטעינה (קובץ חסר) אינו מפיל את היישום: מוחזר QIcon ריק."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

APP_USER_MODEL_ID = "FixPic.App"


def assets_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "fixpic" / "ui" / "assets"
    return Path(__file__).resolve().parent / "assets"


def app_icon() -> QIcon:
    d = assets_dir()
    for name in ("fixpic.ico", "fixpic.png"):
        p = d / name
        if p.is_file():
            icon = QIcon(str(p))
            if not icon.isNull():
                return icon
    return QIcon()


def set_windows_app_id() -> None:
    """ב-Windows, בלי מזהה משלו סרגל המשימות מקבץ את החלון תחת python.exe ומציג את האייקון שלו."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:  # noqa: BLE001 - קוסמטי בלבד
        pass
