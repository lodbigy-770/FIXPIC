"""Canvas (ARCHITECTURE §13, M1.5 UI Spike): תצוגת תמונה עם Zoom/Pan חלק גם על 24MP.

התמונה מוחזקת כפירמידה בזיכרון (רמה k = 1/2^k, INTER_AREA), וב-`paintEvent` מצויר רק מלבן המקור הנראה מהרמה
המתאימה. הפירמידה נבנתה ונבדקה ב-`tools/m1/ui_spike.py` (הוחלט ב-M1.5: ללא Tiles).

חוזים:
- מצב View: `scale` (פיקסלי מסך לפיקסל תמונה) ו-`offset` (נקודת תמונה -> מסך: `p * scale + offset`).
- "מצב התאמה": כל עוד המשתמש לא הזיז או הגדיל, שינוי גודל החלון מתאים מחדש את התמונה.
- ברירת מחדל: גרירה בכפתור שמאלי מזיזה (`left_drag_pans`). במצב ציור (`set_painting(True)`, M3.4) הכפתור השמאלי
  מצייר, הימני מוחק, והזזה = כפתור אמצעי או Space+גרירה.
- **משיכות (M3.4):** ה-Canvas אינו נוגע במסכה. הוא משדר `strokeStarted(נקודה, מחיקה)`, `strokeMoved(נקודה)`,
  `strokeEnded()` בקואורדינטות תמונה של **מרכזי פיקסלים** (פיקסל i = מרכזו ב-i, כמו `MaskLayer.paint`), והצד
  המחבר מצייר ב-`MaskLayer` ואז קורא ל-`update()`.
- **שכבת המסכה** (`set_mask_layer`) ומשיכות בעיבוד (`set_pending_source`) מצוירות מעל התמונה ברזולוציית המסך בלבד:
  חלון המסכה הנראה מוקטן (INTER_AREA; בהגדלה: NEAREST) ל-QImage במידות המסך עם טבלת צבעים. התוצאה נשמרת במטמון לפי
  גרסת המסכה והתצוגה, כך שריחוף העכבר אינו מחשב אותה מחדש.
- תמונה עם Alpha מוצגת על רקע אפור-בהיר (תצוגה בלבד; ה-Alpha עצמו נשמר ב-Session).
- בניית פירמידה ב-thread של ה-UI: נמדד כ-0.05 שניות ל-24MP בספייק.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..imgio import composite_on
from ..strings import t

MIN_SCALE, MAX_SCALE = 0.02, 32.0
ZOOM_STEP = 1.25
KEEP_VISIBLE = 48           # לפחות כך וכך פיקסלי מסך מהתמונה חייבים להישאר בתוך החלון
TRANSPARENT_BG = (204, 204, 204)
MASK_COLOR = (255, 40, 60)        # המסכה הנוכחית (אדום)
PENDING_COLOR = (40, 120, 255)    # משיכות בעיבוד (כחול)
OVERLAY_OPACITY = 140             # 0..255 במסכה מלאה


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


def _color_table(rgb: tuple[int, int, int], opacity: int) -> list[int]:
    """טבלת צבעים של 256 ערכים: ערך מסכה i -> צבע עם שקיפות i*opacity/255 (ARGB, לא premultiplied)."""
    return [QColor(rgb[0], rgb[1], rgb[2], i * opacity // 255).rgba() for i in range(256)]


class Canvas(QWidget):
    scaleChanged = Signal(float)
    strokeStarted = Signal(QPointF, bool)     # נקודת תמונה (מרכז פיקסל), האם מחיקה
    strokeMoved = Signal(QPointF)
    strokeEnded = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pyr: Pyramid | None = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.left_drag_pans = True
        self.painting = False                 # מצב ציור (M3.4)
        self.erase_mode = False               # כפתור שמאלי מוחק (במקום לצייר)
        self.brush_diameter = 20.0            # פיקסלי תמונה, לעיגול המברשת בלבד
        self.mask_layer = None                # MaskLayer | None
        self._pending_source: Callable[[], list[np.ndarray]] | None = None
        self._fit_mode = True
        self._drag: QPointF | None = None
        self._space = False
        self._hover: QPointF | None = None
        self._stroking: int | None = None     # הכפתור שפתח את המשיכה הפעילה
        self._stroke_erase = False
        self._ov_cache: dict | None = None    # {"key", "items": [(QImage, QRectF, array)]}
        self._pend_cache: tuple | None = None # (refs, union)
        self._tables = {MASK_COLOR: _color_table(MASK_COLOR, OVERLAY_OPACITY),
                        PENDING_COLOR: _color_table(PENDING_COLOR, OVERLAY_OPACITY)}
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------ ציור ומסכה
    def set_painting(self, on: bool) -> None:
        """מצב ציור: הכפתור השמאלי מצייר (ולא מזיז). כיבוי באמצע משיכה מסיים אותה."""
        if not on and self._stroking is not None:
            self._end_stroke()
        self.painting = on
        self.left_drag_pans = not on
        self._apply_cursor()
        self.update()

    def set_mask_layer(self, layer) -> None:
        """`MaskLayer` שמצויר מעל התמונה (None = ללא)."""
        self.mask_layer = layer
        self._ov_cache = None
        self.update()

    def set_pending_source(self, fn: Callable[[], list[np.ndarray]] | None) -> None:
        """פונקציה שמחזירה את מסכות המשיכות בעיבוד (`Session.pending_masks`), לציור בצבע נפרד."""
        self._pending_source = fn
        self._ov_cache = None
        self.update()

    def set_brush_diameter(self, d: float) -> None:
        self.brush_diameter = max(1.0, float(d))
        self.update()

    def invalidate_overlay(self) -> None:
        """מחשב מחדש את שכבות המסכה בציור הבא (אחרי שינוי שגרסת השכבה לא משקפת, למשל סיום עיבוד)."""
        self._ov_cache = None
        self.update()

    def _apply_cursor(self) -> None:
        if self._drag is not None:
            self.setCursor(Qt.ClosedHandCursor)
        elif self._space:
            self.setCursor(Qt.OpenHandCursor)
        elif self.painting and self.pyr is not None:
            self.setCursor(Qt.BlankCursor)     # העיגול מצויר ב-paintEvent
        else:
            self.unsetCursor()

    def _pending_union(self) -> np.ndarray | None:
        masks = self._pending_source() if self._pending_source is not None else []
        if not masks:
            self._pend_cache = None
            return None
        c = self._pend_cache
        if c is not None and len(c[0]) == len(masks) and all(a is b for a, b in zip(c[0], masks)):
            return c[1]
        union = masks[0] if len(masks) == 1 else np.bitwise_or.reduce(masks)
        self._pend_cache = (list(masks), union)       # מחזיק הפניות: זהות המערכים לא נעשית ישנה
        return union

    def _overlay_items(self, vis: QRectF) -> list[tuple[QImage, QRectF, np.ndarray]]:
        """תמונות-השכבה לציור (ממוטמן). כל פריט: (QImage ברזולוציית מסך, מלבן יעד במסך, המערך שמאחורי ה-QImage)."""
        layer = self.mask_layer
        pend = self._pending_union()
        if (layer is None or layer.empty) and pend is None:
            return []
        key = (layer.version if layer is not None else -1, id(pend), self.scale, self.offset.x(), self.offset.y(),
               self.width(), self.height())
        if self._ov_cache is not None and self._ov_cache["key"] == key:
            return self._ov_cache["items"]
        items = []
        for arr, color in ((pend, PENDING_COLOR),
                           (None if layer is None or layer.empty else layer.array, MASK_COLOR)):
            if arr is None:
                continue
            it = self._render_overlay(arr, color, vis)
            if it is not None:
                items.append(it)
        self._ov_cache = {"key": key, "items": items, "refs": pend}
        return items

    def _render_overlay(self, mask: np.ndarray, color: tuple[int, int, int], vis: QRectF):
        h, w = mask.shape
        x0, y0 = max(0, math.floor(vis.left())), max(0, math.floor(vis.top()))
        x1, y1 = min(w, math.ceil(vis.right())), min(h, math.ceil(vis.bottom()))
        if x0 >= x1 or y0 >= y1:
            return None
        win = mask[y0:y1, x0:x1]
        ow, oh = max(1, round((x1 - x0) * self.scale)), max(1, round((y1 - y0) * self.scale))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_NEAREST
        small = cv2.resize(win, (ow, oh), interpolation=interp)
        stride = (ow + 3) & ~3                                   # QImage דורש שורות מיושרות ל-4 בתים
        buf = np.zeros((oh, stride), np.uint8)
        buf[:, :ow] = small
        img = QImage(buf.data, ow, oh, stride, QImage.Format_Indexed8)
        img.setColorTable(self._tables[color])
        dst = QRectF(x0 * self.scale + self.offset.x(), y0 * self.scale + self.offset.y(),
                     (x1 - x0) * self.scale, (y1 - y0) * self.scale)
        return img, dst, buf

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
        self._ov_cache = None
        self._apply_cursor()
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
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            for img, rect, _keep in self._overlay_items(vis):    # `_keep` מחזיק את הזיכרון שמאחורי ה-QImage
                p.drawImage(rect, img)
        self._paint_brush_cursor(p)
        p.end()

    def _paint_brush_cursor(self, p: QPainter) -> None:
        if not self.painting or self._hover is None or self._space or self._drag is not None:
            return
        r = max(1.5, self.brush_diameter * self.scale / 2)
        c = self._hover
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 200), 3))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.5 if not self.erase_mode else 1.5, Qt.DashLine if self.erase_mode
                      else Qt.SolidLine))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1))
        p.drawPoint(c)

    # ----------------------------------------------------------------- קלט
    def wheelEvent(self, e) -> None:
        dy = e.angleDelta().y()
        if dy:
            self.zoom_at(e.position(), ZOOM_STEP ** (dy / 120))
        e.accept()

    def _pan_button(self, e) -> bool:
        return (e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and (self.left_drag_pans or self._space)))

    def _stroke_point(self, e) -> QPointF:
        """נקודת תמונה במרכזי פיקסלים (פיקסל i מכסה [i, i+1) במערכת התמונה, ולכן מרכזו ב-i+0.5)."""
        p = self.view_to_image(e.position())
        return QPointF(p.x() - 0.5, p.y() - 0.5)

    def mousePressEvent(self, e) -> None:
        if self.pyr is None:
            super().mousePressEvent(e)
        elif self._pan_button(e):
            self._drag = e.position()
            self._apply_cursor()
            e.accept()
        elif self.painting and self._stroking is None and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._stroking = int(e.button().value)
            erase = e.button() == Qt.RightButton or self.erase_mode
            self._stroke_erase = erase
            self.strokeStarted.emit(self._stroke_point(e), erase)
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        self._hover = e.position()
        if self._drag is not None:
            self.pan_by(e.position() - self._drag)
            self._drag = e.position()
            e.accept()
        elif self._stroking is not None:
            self.strokeMoved.emit(self._stroke_point(e))
            self.update()
            e.accept()
        elif self.painting:
            self.update()                # עיגול המברשת עוקב אחרי העכבר

    def mouseReleaseEvent(self, e) -> None:
        if self._drag is not None and (e.button() == Qt.MiddleButton or e.button() == Qt.LeftButton):
            self._drag = None
            self._apply_cursor()
            e.accept()
        elif self._stroking is not None and int(e.button().value) == self._stroking:
            self._end_stroke()
            e.accept()

    def _end_stroke(self) -> None:
        self._stroking = None
        self.strokeEnded.emit()
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = None
        self.update()
        super().leaveEvent(e)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._apply_cursor()
            self.update()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._apply_cursor()
            self.update()
        else:
            super().keyReleaseEvent(e)

    def resizeEvent(self, _) -> None:
        if self._fit_mode:
            self.fit()
        else:
            self.offset = self._clamped(self.scale, self.offset)
