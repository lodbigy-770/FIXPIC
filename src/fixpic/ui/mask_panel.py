"""לוח כלי מסכה (M4, ממשק): הרחבה/כיווץ, מחיקת רכיבים קטנים, ניקוי, Undo/Redo של עריכות מסכה.

חוזים:
- הלוח עובד על `MaskLayer` אחד שמחברים ב-`bind` (ה-`Session.mask` של התמונה הנוכחית) ויוצר לו `MaskHistory` משלו.
  `bind(None)` מנתק ומשבית. אין תלות ב-`Canvas`: הלוח רק משנה את השכבה ומשדר `maskChanged(rect)` (מלבן ששונה,
  `(x0, y0, x1, y1)`) כדי שהצד המצייר יצייר מחדש.
- כל פעולה היא צעד אחד ב-`MaskHistory.do`; פעולה שלא שינתה דבר אינה משדרת ואינה נרשמת.
- `remove_component_at(x, y)` מיועדת לקריאה מה-Canvas (לחיצה על רכיב) ב-M3.4/M4.
- כשהחלון עסוק (`set_enabled_for_work(False)`) הכפתורים מושבתים והפעולות לא רצות.
- שגיאות `FixPicError` מדווחות ב-`failed(str)` ואינן משנות את השכבה.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QSpinBox, QWidget

from ..errors import FixPicError
from ..session import MaskLayer
from ..session.mask_history import MaskHistory
from ..strings import t

MAX_GROW_PX = 200
DEFAULT_GROW_PX = 3
DEFAULT_MIN_AREA = 50


class MaskToolsPanel(QWidget):
    maskChanged = Signal(object)        # Rect | None
    failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layer: MaskLayer | None = None
        self.history: MaskHistory | None = None
        self._work_ok = True
        self.spin_px = QSpinBox()
        self.spin_px.setRange(1, MAX_GROW_PX)
        self.spin_px.setValue(DEFAULT_GROW_PX)
        self.spin_px.setSuffix(" px")
        self.spin_area = QSpinBox()
        self.spin_area.setRange(1, 1_000_000)
        self.spin_area.setValue(DEFAULT_MIN_AREA)
        self.spin_area.setSuffix(" px²")
        self.btn_grow = QPushButton(t("mask.grow"))
        self.btn_shrink = QPushButton(t("mask.shrink"))
        self.btn_small = QPushButton(t("mask.remove_small"))
        self.btn_clear = QPushButton(t("mask.clear"))
        self.btn_undo = QPushButton(t("mask.undo"))
        self.btn_redo = QPushButton(t("mask.redo"))
        g = QGridLayout(self)
        g.addWidget(QLabel(t("mask.amount")), 0, 0)
        g.addWidget(self.spin_px, 0, 1)
        g.addWidget(self.btn_grow, 1, 0)
        g.addWidget(self.btn_shrink, 1, 1)
        g.addWidget(QLabel(t("mask.min_area")), 2, 0)
        g.addWidget(self.spin_area, 2, 1)
        g.addWidget(self.btn_small, 3, 0, 1, 2)
        g.addWidget(self.btn_clear, 4, 0, 1, 2)
        g.addWidget(self.btn_undo, 5, 0)
        g.addWidget(self.btn_redo, 5, 1)
        g.setRowStretch(6, 1)
        self.btn_grow.clicked.connect(lambda _c=False: self.grow())
        self.btn_shrink.clicked.connect(lambda _c=False: self.shrink())
        self.btn_small.clicked.connect(lambda _c=False: self.remove_small())
        self.btn_clear.clicked.connect(lambda _c=False: self.clear())
        self.btn_undo.clicked.connect(lambda _c=False: self.undo())
        self.btn_redo.clicked.connect(lambda _c=False: self.redo())
        self.refresh()

    # ------------------------------------------------------------------ חיבור
    def bind(self, layer: MaskLayer | None) -> None:
        self.layer = layer
        self.history = MaskHistory(layer) if layer is not None else None
        self.refresh()

    def set_enabled_for_work(self, ok: bool) -> None:
        self._work_ok = ok
        self.refresh()

    @property
    def active(self) -> bool:
        return self.layer is not None and self._work_ok

    def refresh(self) -> None:
        """מעדכן הפעלת כפתורים לפי מצב השכבה וההיסטוריה (לקרוא גם אחרי שינוי מסכה מבחוץ)."""
        on = self.active
        has = on and not self.layer.empty
        h = self.history
        for b in (self.btn_grow, self.btn_shrink, self.btn_small, self.btn_clear):
            b.setEnabled(has)
        self.btn_undo.setEnabled(on and h is not None and h.can_undo)
        self.btn_redo.setEnabled(on and h is not None and h.can_redo)
        for s in (self.spin_px, self.spin_area):
            s.setEnabled(on)

    # ---------------------------------------------------------------- פעולות
    def grow(self, px: int | None = None):
        n = self.spin_px.value() if px is None else px
        return self._edit(lambda layer: layer.grow(n))

    def shrink(self, px: int | None = None):
        n = self.spin_px.value() if px is None else px
        return self._edit(lambda layer: layer.shrink(n))

    def remove_small(self, min_area: int | None = None):
        a = self.spin_area.value() if min_area is None else min_area

        def op(layer: MaskLayer):
            box = layer.bbox()
            return box if layer.remove_small(a) else None
        return self._edit(op)

    def remove_component_at(self, x: float, y: float):
        return self._edit(lambda layer: layer.remove_component_at(x, y))

    def clear(self):
        def op(layer: MaskLayer):
            box = layer.bbox()
            layer.clear()
            return box
        return self._edit(op)

    def undo(self):
        return self._step(lambda: self.history.undo(), lambda: self.history.can_undo)

    def redo(self):
        return self._step(lambda: self.history.redo(), lambda: self.history.can_redo)

    # ---------------------------------------------------------------- פנימי
    def _edit(self, op):
        if not self.active:
            return None
        try:
            rect = self.history.do(op)
        except FixPicError as e:
            self.failed.emit(str(e))
            self.refresh()
            return None
        self._done(rect)
        return rect

    def _step(self, run, can):
        if not self.active or not can():
            return None
        try:
            rect = run()
        except FixPicError as e:
            self.failed.emit(str(e))
            self.refresh()
            return None
        self._done(rect)
        return rect

    def _done(self, rect) -> None:
        self.refresh()
        if rect is not None:
            self.maskChanged.emit(rect)
