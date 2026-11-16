"""M5: מסך "אודות": גרסה, רישיון המודל, והודעות הרישיון של רכיבי צד שלישי (`THIRD_PARTY_NOTICES.txt`).

הקובץ נוצר בזמן האריזה (`tools/m5/third_party_notices.py`) ונארז לצד האפליקציה. בהרצה מארוזה הוא נמצא ב-`sys._MEIPASS` (תיקיית
`_internal` של PyInstaller) או ליד ה-exe; בפיתוח: `build/THIRD_PARTY_NOTICES.txt` בשורש הריפו. `FIXPIC_NOTICES` גובר על כולם."""
from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..strings import t

NOTICES_NAME = "THIRD_PARTY_NOTICES.txt"


def notices_candidates() -> list[Path]:
    """נתיבי החיפוש לפי סדר עדיפות."""
    c: list[Path] = []
    env = os.environ.get("FIXPIC_NOTICES")
    if env:
        c.append(Path(env))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        c.append(Path(meipass) / NOTICES_NAME)
    if getattr(sys, "frozen", False):
        c.append(Path(sys.executable).resolve().parent / NOTICES_NAME)
    c.append(Path(__file__).resolve().parents[3] / "build" / NOTICES_NAME)
    return c


def find_notices_path() -> Path | None:
    for p in notices_candidates():
        if p.is_file():
            return p
    return None


def read_notices() -> tuple[str | None, list[Path]]:
    """(הטקסט או None, הנתיבים שנבדקו). קובץ לא קריא נחשב חסר."""
    cands = notices_candidates()
    for p in cands:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace"), cands
        except OSError:
            continue
    return None, cands


def app_version() -> str:
    try:
        return metadata.version("fixpic")
    except metadata.PackageNotFoundError:       # הרצה מארוזה ללא dist-info
        return "dev"


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.about_title"))
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        self.body = QLabel(t("msg.about_body", version=app_version()))
        self.body.setWordWrap(True)
        lay.addWidget(self.body)
        lay.addWidget(QLabel(t("dialog.notices_label")))
        self.notices = QPlainTextEdit()
        self.notices.setReadOnly(True)
        self.notices.setLayoutDirection(Qt.LeftToRight)         # טקסט הרישיונות באנגלית
        self.notices.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        text, cands = read_notices()
        self.notices_found = text is not None
        self.notices.setPlainText(text if text is not None
                                  else t("msg.notices_missing", paths="\n".join(str(p) for p in cands)))
        lay.addWidget(self.notices, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_close = QPushButton(t("action.close"))
        self.btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
