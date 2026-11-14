"""Canvas (ARCHITECTURE §13, M1.5 UI Spike): תצוגת תמונה עם Zoom/Pan חלק גם על 24MP.

התמונה מוחזקת כפירמידה בזיכרון (רמה k = 1/2^k, INTER_AREA), וב-`paintEvent` מצויר רק מלבן המקור הנראה מהרמה
המתאימה. הפירמידה נבנתה ונבדקה ב-`tools/m1/ui_spike.py` (הוחלט ב-M1.5: ללא Tiles).

חוזים:
- מצב View: `scale` (פיקסלי מסך לפיקסל תמונה) ו-`offset` (נקודת תמונה -> מסך: `p * scale + offset`).
- "מצב התאמה": כל עוד המשתמש לא הזיז או הגדיל, שינוי גודל החלון מתאים מחדש את התמונה.
- ברירת מחדל: גרירה בכפתור שמאלי מזיזה (`left_drag_pans`); M3.4 יכבה זאת כדי שהכפתור השמאלי יצייר, ואז
  הזזה = כפתור אמצעי או Space+גרירה.
- תמונה עם Alpha מוצגת על רקע אפור-בהיר (תצוגה בלבד; ה-Alpha עצמו נשמר ב-Session).
- בניית פירמידה ב-thread של ה-UI: נמדד כ-0.05 שניות ל-24MP בספייק.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..imgio import composite_on
from ..strings import t

MIN_SCALE, MAX_SCALE = 0.02, 32.0
ZOOM_STEP = 1.25
KEEP_VISIBLE = 48           # לפחות כך וכך פיקסלי מסך מהתמונה חייבים להישאר בתוך החלון
TRANSPARENT_BG = (204, 204, 204)


class Pyramid:
    """רמות הקטנה של תמונת RGB. רמה 0 משתפת את המערך המקורי (ללא העתקה)."""

    def __init__(self, rgb: np.ndarray) -> None:
        self.arrays: list[np.ndarray] = []
        self.levels: list[QImage] = []
        cur = np.ascontiguousarray(rgb)
        while True:
            self.arrays.append(cur)
            self.levels.append(QImage(cur.data, cur.shape[1], cur.shape[0], cur.strides[0], QImage.Format_RGB888))
            if min(cur.shape[:2]) <= 512:
                break
            cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
        self.w, self.h = self.levels[0].width(), self.levels[0].height()

    def level_for(self, scale: float) -> int:
        if scale >= 1:
            return 0
        return min(int(math.floor(math.log2(1 / scale))), len(self.levels) - 1)


class Canvas(QWidget):
    scaleChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pyr: Pyramid | None = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.left_drag_pans = True
        self._fit_mode = True
        self._drag: QPointF | None = None
        self._space = False
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ---------------------------------------------------------------- תמונה
    @property
    def has_image(self) -> bool:
        return self.pyr is not None

    def set_image(self, rgb: np.ndarray | None, alpha: np.ndarray | None = None, *, keep_view: bool = False) -> None:
        """מציג תמונה (או מנקה עם None). `keep_view=True` שומר זום והזזה (עדכון אחרי Commit/Undo בגודל זהה)."""
        if rgb is None:
            self.pyr = None
        else:
            shown = rgb if alpha is None else composite_on(rgb, alpha, TRANSPARENT_BG)
            same = self.pyr is not None and (self.pyr.w, self.pyr.h) == (shown.shape[1], shown.shape[0])
            self.pyr = Pyramid(shown)
            if not (keep_view and same):
                self.fit()
        self.update()

    # ----------------------------------------------------------- מתמטיקת תצוגה
    def fit(self) -> None:
        if self.pyr is None:
            return
        self._fit_mode = True
        s = min(self.width() / self.pyr.w, self.height() / self.pyr.h, 1.0)   # לא מגדילים מעבר ל-100% בהתאמה
        self._set_view(s, QPointF((self.width() - self.pyr.w * s) / 2, (self.height() - self.pyr.h * s) / 2))

    def actual_size(self) -> None:
        self.zoom_at(QPointF(self.width() / 2, self.height() / 2), 1 / self.scale)

    def zoom_at(self, pos: QPointF, factor: float) -> None:
        """זום סביב נקודת מסך: הנקודה שמתחת ל-`pos` נשארת במקומה."""
        if self.pyr is None:
            return
        new = min(max(self.scale * factor, MIN_SCALE), MAX_SCALE)
        k = new / self.scale
        self._fit_mode = False
        self._set_view(new, QPointF(pos.x() - (pos.x() - self.offset.x()) * k,
                                    pos.y() - (pos.y() - self.offset.y()) * k))

    def pan_by(self, delta: QPointF) -> None:
        if self.pyr is None:
            return
        self._fit_mode = False
        self._set_view(self.scale, self.offset + delta)

    def image_to_view(self, p: QPointF) -> QPointF:
        return QPointF(p.x() * self.scale + self.offset.x(), p.y() * self.scale + self.offset.y())

    def view_to_image(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale)

    def _set_view(self, scale: float, offset: QPointF) -> None:
        old = self.scale
        self.scale, self.offset = scale, self._clamped(scale, offset)
        if scale != old:
            self.scaleChanged.emit(scale)
        self.update()

    def _clamped(self, scale: float, off: QPointF) -> QPointF:
        """התמונה לא יכולה לצאת כולה מהחלון: נשארים לפחות KEEP_VISIBLE פיקסלים (או כל התמונה אם קטנה מזה)."""
        if self.pyr is None:
            return off
        iw, ih = self.pyr.w * scale, self.pyr.h * scale
        kx, ky = min(KEEP_VISIBLE, iw), min(KEEP_VISIBLE, ih)
        x = min(max(off.x(), kx - iw), self.width() - kx)
        y = min(max(off.y(), ky - ih), self.height() - ky)
        return QPointF(x, y)

    def visible_source_rect(self) -> QRectF:
        """החלק הנראה של התמונה, בפיקסלי תמונה (ריק אם אין תמונה)."""
        if self.pyr is None:
            return QRectF()
        vis = QRectF(-self.offset.x() / self.scale, -self.offset.y() / self.scale,
                     self.width() / self.scale, self.height() / self.scale)
        return vis.intersected(QRectF(0, 0, self.pyr.w, self.pyr.h))

    # ------------------------------------------------------------------ ציור
    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        if self.pyr is None:
            p.setPen(self.palette().windowText().color())
            p.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignCenter | Qt.TextWordWrap, t("hint.empty"))
            p.end()
            return
        vis = self.visible_source_rect()
        if not vis.isEmpty():
            lvl = self.pyr.level_for(self.scale)
            f = 1 / (2 ** lvl)
            src = QRectF(vis.x() * f, vis.y() * f, vis.width() * f, vis.height() * f)
            dst = QRectF(vis.x() * self.scale + self.offset.x(), vis.y() * self.scale + self.offset.y(),
                         vis.width() * self.scale, vis.height() * self.scale)
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2)
            p.drawImage(dst, self.pyr.levels[lvl], src)
        p.end()

    # ----------------------------------------------------------------- קלט
    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if dy:
            self.zoom_at(e.position(), ZOOM_STEP ** (dy / 120))
        e.accept()

    def _pan_button(self, e) -> bool:
        return (e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and (self.left_drag_pans or self._space)))

    def mousePressEvent(self, e) -> None:
        if self._pan_button(e) and self.pyr is not None:
            self._drag = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self.pan_by(e.position() - self._drag)
            self._drag = e.position()
            e.accept()

    def mouseReleaseEvent(self, e) -> None:
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            e.accept()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            if self._drag is None:
                self.unsetCursor()
        else:
            super().keyReleaseEvent(e)

    def resizeEvent(self, _) -> None:
        if self._fit_mode:
            self.fit()
        else:
            self.offset = self._clamped(self.scale, self.offset)
