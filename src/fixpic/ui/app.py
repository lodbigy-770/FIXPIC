"""נקודת כניסה לממשק: `python -m fixpic.app [קובץ תמונה]`. QApplication ב-RTL, חלון ראשי, פתיחת קובץ מהשורה."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ..strings import t
from .engine import default_engine_factory
from .main_window import MainWindow


def create_app(argv: list[str] | None = None) -> QApplication:
    """QApplication קיים או חדש, עם כיוון RTL ושם האפליקציה (עברית בלבד, PARITY-SPEC §2)."""
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setLayoutDirection(Qt.RightToLeft)
    app.setApplicationName(t("app.name"))
    return app


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = create_app(argv)
    win = MainWindow(engine_factory=default_engine_factory())
    win.show()
    if len(argv) > 1:
        win.open_path(argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
