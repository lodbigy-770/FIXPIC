"""M5: הודעת שגיאה חמורה (חריגה שלא נתפסה) מכל thread. ה-hook של `diagnostics` רץ ב-thread שבו קרתה החריגה, ולכן
ההודעה עוברת ב-Signal ל-thread של הממשק. בזמן שהודעה פתוחה הודעות נוספות לא נערמות (הדוחות עצמם נכתבים תמיד)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from ..strings import t


class CrashNotifier(QObject):
    _crashed = Signal(str)          # נתיב הדוח, או "" אם לא נכתב

    def __init__(self, parent_widget: QWidget | None = None) -> None:
        super().__init__()
        self.parent_widget = parent_widget
        self.shown = 0
        self._open = False
        self._crashed.connect(self._show)

    def notify(self, path: Path | None) -> None:
        """מתאים ל-`install_crash_handlers(on_crash=...)`; בטוח לקריאה מכל thread."""
        self._crashed.emit(str(path) if path else "")

    def _show(self, path: str) -> None:
        if self._open:
            return
        self._open = True
        try:
            self.shown += 1
            text = t("error.crash", path=path) if path else t("error.crash_nopath")
            QMessageBox.critical(self.parent_widget, t("error.crash_title"), text)
        finally:
            self._open = False
