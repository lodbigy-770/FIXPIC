"""עזרי התאמה למסכים קטנים (למשל 14 אינץ'): לוחות כפתורים בגלילה וגודל חלון לפי שטח המסך הפנוי."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QWidget

MIN_WINDOW = QSize(640, 420)          # מינימום חלון: אפשר להקטין עד כאן, והלוחות נגללים
PREFERRED_WINDOW = QSize(1100, 760)
SCREEN_FRACTION = 0.92                # חלון לא יתפוס יותר מחלק זה משטח המסך הפנוי


class PanelScroll(QScrollArea):
    """עוטף לוח כפתורים בגלילה אנכית: הלוח לא מכריח גובה מינימלי על החלון, והגלילה מופיעה רק כשצריך.
    `panel` הוא הלוח המקורי (נשמר כדי שהקוד והבדיקות ימשיכו לפנות אליו ישירות)."""

    def __init__(self, panel: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.setWidget(panel)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)   # גובה גמיש: לא מנפח את החלון
        self.setMinimumHeight(96)

    def sizeHint(self) -> QSize:
        h = self.panel.sizeHint()
        bar = self.verticalScrollBar().sizeHint().width()
        return QSize(h.width() + bar + 4, min(h.height(), 360))

    def minimumSizeHint(self) -> QSize:
        return QSize(self.panel.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width() + 4, 96)


def initial_window_size(available: QSize | None) -> QSize:
    """גודל פתיחה: 1100x760 כשיש מקום, אחרת חלק מהשטח הפנוי של המסך (לא פחות מהמינימום)."""
    if available is None or available.isEmpty():
        return PREFERRED_WINDOW
    w = min(PREFERRED_WINDOW.width(), int(available.width() * SCREEN_FRACTION))
    h = min(PREFERRED_WINDOW.height(), int(available.height() * SCREEN_FRACTION))
    return QSize(max(MIN_WINDOW.width(), w), max(MIN_WINDOW.height(), h))
