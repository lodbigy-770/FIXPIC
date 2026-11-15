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
- **השוואה לפני/אחרי (M3.5):** `set_compare(on, original_rgb, original_alpha, badge)` מחליף את התצוגה בתמונה המקורית
  (פירמידה נבנית בפעם הראשונה ונשמרת עד `clear_compare_cache`), בלי שכבת המסכה ובלי עיגול מברשת, עם תג בפינה. הזום
  וההזזה נשארים, כך שהמעבר ללא קפיצה. מקש B (`compareHoldChanged`) = החזקה; אובדן פוקוס משחרר אותה.
- **שכבת המסכה** (`set_mask_layer`) ומשיכות בעיבוד (`set_pending_source`) מצוירות מעל התמונה ברזולוציית המסך בלבד:
  חלון המסכה הנראה מוקטן (INTER_AREA; בהגדלה: NEAREST) ל-QImage במידות המסך עם טבלת צבעים. התוצאה נשמרת במטמון לפי
  גרסת המסכה והתצוגה, כך שריחוף העכבר אינו מחשב אותה מחדש.
- תמונה עם Alpha מוצגת על רקע אפור-בהיר (תצוגה בלבד; ה-Alpha עצמו נשמר ב-Session).
- בניית פירמידה ב-thread של ה-UI: נמדד כ-0.05 שניות ל-24MP בספייק.
- **בחירה חכמה (M4, שלב 4ב):** `set_select_mode("box"|"lasso"|None)` מחליף את משמעות הכפתור השמאלי במצב ציור: גרירה מציירת
  מלבן/לאסו, ובשחרור משודר `selectBoxDrawn((x0, y0, x1, y1))` או `selectLassoDrawn([(x, y), ...])` בפיקסלי תמונה (לא חתוכים).
  כש-`set_select_scribble(True)` (יש הצעה/קלט פעיל) גרירה שמאלית = משיכה מתקנת חיובית, ימנית או Alt = שלילית
  (`selectScribbleDrawn(path, positive)`), ו-Shift+גרירה שמאלית מתחילה מלבן/לאסו חדש. ה-Canvas אינו נוגע במסכה:
  `set_proposal(Proposal | None)` מצייר את ההצעה בירוק מעל התמונה (חלון קטן במקומו, בלי מסכה בגודל התמונה). `cancel_gesture()`
  מבטל מחוות פעילה (Esc). כל זה פעיל רק במצב ציור; בלעדיו (עסוק/השוואה) כפתור שמאל מזיז כרגיל.
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
SELECT_COLOR = (40, 200, 90)      # הצעת בחירה חכמה (ירוק)
SCRIBBLE_NEG_COLOR = (255, 140, 0)  # משיכה מתקנת שלילית (כתום), בזמן המשיכה בלבד
OVERLAY_OPACITY = 140             # 0..255 במסכה מלאה
MIN_GESTURE_PX = 4                # מלבן קטן מזה במסך נחשב ללחיצה ולא משודר
GESTURE_STEP_PX = 2               # מרחק מינימלי במסך בין נקודות לאסו/משיכה
PAN_STEP = 60                     # פיקסלי מסך להזזה בגלגלת (Shift/Alt)


def _half(a: np.ndarray, w2: int, h2: int) -> np.ndarray:
    """הקטנה פי 2 מדויקת: כל פיקסל ברמה הבאה = ממוצע 2x2 (שורה/עמודה אחרונה אי-זוגית נחתכת). המדויקות מאפשרת עדכון
    אזורי של הפירמידה (`Pyramid.update_region`) שזהה לבייט לבנייה מחדש."""
    return cv2.resize(a[:h2 * 2, :w2 * 2], (w2, h2), interpolation=cv2.INTER_AREA)


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
            cur = _half(cur, cur.shape[1] // 2, cur.shape[0] // 2)
        self.w, self.h = self.levels[0].width(), self.levels[0].height()

    def update_region(self, level0: np.ndarray, box: tuple[int, int, int, int]) -> None:
        """מחליף את רמה 0 ב-`level0` (אותן מידות) ומחשב מחדש בכל רמה רק את האזור שמכסה את `box` (x0, y0, x1, y1).
        רמות 1 ומעלה נכתבות במקום (הזיכרון שלהן שייך לפירמידה); רמה 0 נשארת משותפת עם `level0`."""
        if level0.shape[:2] != self.arrays[0].shape[:2]:
            raise ValueError("update_region: מידות שונות")
        cur = np.ascontiguousarray(level0)
        self.arrays[0] = cur
        self.levels[0] = QImage(cur.data, cur.shape[1], cur.shape[0], cur.strides[0], QImage.Format_RGB888)
        x0, y0, x1, y1 = box
        for k in range(1, len(self.arrays)):
            dst, prev = self.arrays[k], self.arrays[k - 1]
            h2, w2 = dst.shape[:2]
            x0, y0, x1, y1 = x0 // 2, y0 // 2, min(w2, -(-x1 // 2)), min(h2, -(-y1 // 2))
            if x0 >= x1 or y0 >= y1:
                break
            dst[y0:y1, x0:x1] = cv2.resize(prev[2 * y0:2 * y1, 2 * x0:2 * x1], (x1 - x0, y1 - y0),
                                           interpolation=cv2.INTER_AREA)

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
    compareHoldChanged = Signal(bool)         # מקש ההחזקה (B) נלחץ/שוחרר
    selectBoxDrawn = Signal(object)           # (x0, y0, x1, y1) בפיקסלי תמונה, לא חתוך, x0<x1 ו-y0<y1
    selectLassoDrawn = Signal(object)         # [(x, y), ...] בפיקסלי תמונה
    selectScribbleDrawn = Signal(object, bool)  # ([(x, y), ...], positive)

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
        self.select_mode: str | None = None   # None | "box" | "lasso" (בחירה חכמה)
        self.select_scribble = False          # יש קלט פעיל: גרירה = משיכה מתקנת
        self._proposal = None                 # Proposal | None, מצויר בירוק
        self._gesture: dict | None = None     # {"kind", "button", "pts": [QPointF תמונה], "positive"}
        self._comparing = False
        self._cmp_pyr: Pyramid | None = None
        self._cmp_badge = ""
        self._hold_key = False
        self._ov_cache: dict | None = None    # {"key", "items": [(QImage, QRectF, array)]}
        self._pend_cache: tuple | None = None # (refs, union)
        self._tables = {MASK_COLOR: _color_table(MASK_COLOR, OVERLAY_OPACITY),
                        PENDING_COLOR: _color_table(PENDING_COLOR, OVERLAY_OPACITY),
                        SELECT_COLOR: _color_table(SELECT_COLOR, OVERLAY_OPACITY)}
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------ ציור ומסכה
    def set_painting(self, on: bool) -> None:
        """מצב ציור: הכפתור השמאלי מצייר (ולא מזיז). כיבוי באמצע משיכה מסיים אותה."""
        if not on and self._stroking is not None:
            self._end_stroke()
        if not on:
            self.cancel_gesture()
        self.painting = on
        self.left_drag_pans = not on
        self._apply_cursor()
        self.update()

    # ------------------------------------------------------------ בחירה חכמה
    def set_select_mode(self, mode: str | None) -> None:
        """`"box"` / `"lasso"` = כפתור שמאל מצייר מלבן/לאסו (במצב ציור); None = מברשת רגילה."""
        if mode not in (None, "box", "lasso"):
            raise ValueError(f"מצב בחירה לא מוכר: {mode}")
        self.cancel_gesture()
        self.select_mode = mode
        self._apply_cursor()
        self.update()

    def set_select_scribble(self, on: bool) -> None:
        """יש קלט בחירה פעיל: גרירה היא משיכה מתקנת (ולא מלבן/לאסו חדש)."""
        if on != self.select_scribble:
            self.cancel_gesture()
            self.select_scribble = on
            self._apply_cursor()
            self.update()

    def set_proposal(self, proposal) -> None:
        """הצעת בחירה (`select.Proposal`) לציור בירוק מעל התמונה, או None. אינה נוגעת במסכה."""
        if proposal is not self._proposal:
            self._proposal = proposal
            self._ov_cache = None
            self.update()

    @property
    def gesture_active(self) -> bool:
        return self._gesture is not None

    def cancel_gesture(self) -> bool:
        """מבטל מחוות בחירה פעילה (מלבן/לאסו/משיכה) בלי לשדר. True אם הייתה כזו."""
        if self._gesture is None:
            return False
        self._gesture = None
        self.update()
        return True

    @property
    def comparing(self) -> bool:
        return self._comparing

    def set_compare(self, on: bool, rgb: np.ndarray | None = None, alpha: np.ndarray | None = None,
                    badge: str = "") -> None:
        """מציג את התמונה המקורית (`rgb`, `alpha`) במקום העובדת. הפירמידה נשמרת במטמון בין הפעלות."""
        if on and self.pyr is not None:
            if self._cmp_pyr is None:
                if rgb is None:
                    on = False
                else:
                    shown = rgb if alpha is None else composite_on(rgb, alpha, TRANSPARENT_BG)
                    self._cmp_pyr = Pyramid(shown)
            if self._cmp_pyr is not None and (self._cmp_pyr.w, self._cmp_pyr.h) != (self.pyr.w, self.pyr.h):
                on = False                                          # מידות שונות: אין השוואה (לא קורה היום)
        else:
            on = False
        self._comparing, self._cmp_badge = on, badge
        self._apply_cursor()
        self.update()

    def update_image_region(self, rgb: np.ndarray, alpha: np.ndarray | None, box: tuple[int, int, int, int] | None) -> None:
        """מרענן אחרי שינוי בתמונה העובדת באזור `box` בלבד (Commit/Undo/Redo): הרכבת ה-Alpha ועדכון רמות הפירמידה רק
        שם, במקום בנייה מחדש של כל התמונה (בתמונה עם Alpha של 24MP זה ההבדל בין כ-10ms לכ-600ms). בלי `box`, או כשהמבנה
        אינו מתאים, חוזר לבנייה מלאה. הזום וההזזה נשמרים."""
        pyr = self.pyr
        ok = (box is not None and pyr is not None and (pyr.w, pyr.h) == (rgb.shape[1], rgb.shape[0]))
        if ok and alpha is not None:
            a0 = pyr.arrays[0]
            ok = a0.flags.writeable and a0 is not rgb
        if not ok:
            self.set_image(rgb, alpha, keep_view=True)
            return
        x0, y0, x1, y1 = box
        if alpha is None:
            pyr.update_region(rgb, box)
        else:
            a0 = pyr.arrays[0]
            a0[y0:y1, x0:x1] = composite_on(rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], TRANSPARENT_BG)
            pyr.update_region(a0, box)
        self._ov_cache = None
        self.update()

    def clear_compare_cache(self) -> None:
        """משחרר את פירמידת המקור (תמונה חדשה); יוצא ממצב השוואה."""
        self._cmp_pyr = None
        self._comparing = False
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
        elif self.painting and self.pyr is not None and not self._comparing:
            if self.select_mode is not None and not self.select_scribble:
                self.setCursor(Qt.CrossCursor)     # מלבן/לאסו חדש
            else:
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
        prop = self._proposal
        if (layer is None or layer.empty) and pend is None and prop is None:
            return []
        key = (layer.version if layer is not None else -1, id(pend), id(prop), self.scale, self.offset.x(),
               self.offset.y(), self.width(), self.height())
        if self._ov_cache is not None and self._ov_cache["key"] == key:
            return self._ov_cache["items"]
        items = []
        prop_origin = (prop.rect[0], prop.rect[1]) if prop is not None else (0, 0)
        for arr, color, origin in ((pend, PENDING_COLOR, (0, 0)),
                                   (None if layer is None or layer.empty else layer.array, MASK_COLOR, (0, 0)),
                                   (prop.window if prop is not None else None, SELECT_COLOR, prop_origin)):
            if arr is None:
                continue
            it = self._render_overlay(arr, color, vis, origin)
            if it is not None:
                items.append(it)
        self._ov_cache = {"key": key, "items": items, "refs": (pend, prop)}   # מחזיק הפניות: id לא ייעשה ישן
        return items

    def _render_overlay(self, mask: np.ndarray, color: tuple[int, int, int], vis: QRectF,
                        origin: tuple[int, int] = (0, 0)):
        """`mask` ממוקם בפינה `origin` (x, y) של התמונה (חלון הצעה); ברירת מחדל = מסכה בגודל התמונה."""
        h, w = mask.shape
        ox, oy = origin
        x0, y0 = max(ox, math.floor(vis.left())), max(oy, math.floor(vis.top()))
        x1, y1 = min(ox + w, math.ceil(vis.right())), min(oy + h, math.ceil(vis.bottom()))
        if x0 >= x1 or y0 >= y1:
            return None
        win = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
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
        pyr = self._cmp_pyr if self._comparing and self._cmp_pyr is not None else self.pyr
        if not vis.isEmpty():
            lvl = pyr.level_for(self.scale)
            f = 1 / (2 ** lvl)
            src = QRectF(vis.x() * f, vis.y() * f, vis.width() * f, vis.height() * f)
            dst = QRectF(vis.x() * self.scale + self.offset.x(), vis.y() * self.scale + self.offset.y(),
                         vis.width() * self.scale, vis.height() * self.scale)
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 2)
            p.drawImage(dst, pyr.levels[lvl], src)
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            if not self._comparing:
                for img, rect, _keep in self._overlay_items(vis):    # `_keep` מחזיק את הזיכרון שמאחורי ה-QImage
                    p.drawImage(rect, img)
        if self._comparing:
            self._paint_badge(p)
        else:
            self._paint_gesture(p)
            self._paint_brush_cursor(p)
        p.end()

    def _paint_badge(self, p: QPainter) -> None:
        if not self._cmp_badge:
            return
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(self._cmp_badge) + 24, fm.height() + 12
        r = QRectF(self.width() - w - 12, 12, w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QColor(255, 255, 255))
        p.drawText(r, Qt.AlignCenter, self._cmp_badge)

    def _paint_gesture(self, p: QPainter) -> None:
        g = self._gesture
        if g is None or not g["pts"]:
            return
        pts = [self.image_to_view(q) for q in g["pts"]]
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        if g["kind"] == "scribble":
            col = QColor(*(SELECT_COLOR if g["positive"] else SCRIBBLE_NEG_COLOR), 150)
            pen = QPen(col, max(2.0, self.brush_diameter * self.scale), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            if len(pts) == 1:
                p.drawPoint(pts[0])
            else:
                p.drawPolyline(pts)
            return
        for pen in (QPen(QColor(0, 0, 0, 200), 3), QPen(QColor(255, 255, 255, 240), 1.5, Qt.DashLine)):
            p.setPen(pen)
            if g["kind"] == "box":
                if len(pts) >= 2:
                    p.drawRect(QRectF(pts[0], pts[-1]).normalized())
            else:
                p.drawPolyline(pts)
                if len(pts) >= 3:
                    p.drawLine(pts[-1], pts[0])

    def _paint_brush_cursor(self, p: QPainter) -> None:
        if not self.painting or self._hover is None or self._space or self._drag is not None:
            return
        if self.select_mode is not None and not self.select_scribble:
            return                                       # מלבן/לאסו: סמן צלב, בלי עיגול מברשת
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
        """גלגלת = זום סביב הסמן. Shift+גלגלת = הזזה אופקית, Alt+גלגלת = הזזה אנכית, וגלגלת אופקית (משטח מגע) מזיזה
        אופקית. צעד הזזה: 60 פיקסלי מסך לכל "שן" (120)."""
        d, mods = e.angleDelta(), e.modifiers()
        if d.x():
            self.pan_by(QPointF(d.x() / 120 * PAN_STEP, 0))
        elif d.y() and (mods & Qt.ShiftModifier):
            self.pan_by(QPointF(d.y() / 120 * PAN_STEP, 0))
        elif d.y() and (mods & Qt.AltModifier):
            self.pan_by(QPointF(0, d.y() / 120 * PAN_STEP))
        elif d.y():
            self.zoom_at(e.position(), ZOOM_STEP ** (d.y() / 120))
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
        elif (self.painting and self.select_mode is not None and self._gesture is None
              and e.button() in (Qt.LeftButton, Qt.RightButton)):
            self._begin_select_gesture(e)
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
        elif self._gesture is not None:
            self._extend_select_gesture(e)
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
        elif self._gesture is not None and int(e.button().value) == self._gesture["button"]:
            self._finish_select_gesture(e)
            e.accept()
        elif self._stroking is not None and int(e.button().value) == self._stroking:
            self._end_stroke()
            e.accept()

    def _begin_select_gesture(self, e) -> None:
        left = e.button() == Qt.LeftButton
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        if self.select_scribble and not (left and shift):
            positive = left and not (e.modifiers() & Qt.AltModifier)
            kind = "scribble"
        elif left:
            positive, kind = True, self.select_mode
        else:
            super().mousePressEvent(e)            # כפתור ימני בלי קלט פעיל: אין משמעות
            return
        self._gesture = {"kind": kind, "button": int(e.button().value), "positive": positive,
                         "pts": [self.view_to_image(e.position())], "last": e.position()}
        e.accept()
        self.update()

    def _extend_select_gesture(self, e) -> None:
        g = self._gesture
        pos = e.position()
        if g["kind"] == "box":
            g["pts"] = g["pts"][:1] + [self.view_to_image(pos)]
        else:
            d = pos - g["last"]
            if d.x() * d.x() + d.y() * d.y() >= GESTURE_STEP_PX ** 2:
                g["pts"].append(self.view_to_image(pos))
                g["last"] = pos
        self.update()

    def _finish_select_gesture(self, e) -> None:
        g, self._gesture = self._gesture, None
        pts = g["pts"]
        if g["kind"] != "box" and len(pts) >= 1:
            end = self.view_to_image(e.position())
            if pts[-1] != end:
                pts = pts + [end]
        self.update()
        if g["kind"] == "box":
            a = self.image_to_view(pts[0])
            b = self.image_to_view(self.view_to_image(e.position()))
            if abs(a.x() - b.x()) < MIN_GESTURE_PX and abs(a.y() - b.y()) < MIN_GESTURE_PX:
                return                              # לחיצה בלי גרירה: לא מלבן
            q = self.view_to_image(e.position())
            self.selectBoxDrawn.emit((min(pts[0].x(), q.x()), min(pts[0].y(), q.y()),
                                      max(pts[0].x(), q.x()), max(pts[0].y(), q.y())))
        elif g["kind"] == "lasso":
            if len(pts) >= 3:
                self.selectLassoDrawn.emit([(q.x(), q.y()) for q in pts])
        else:
            self.selectScribbleDrawn.emit([(q.x(), q.y()) for q in pts], bool(g["positive"]))

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
        elif e.key() == Qt.Key_B and not e.modifiers() and not e.isAutoRepeat():
            self._hold_key = True
            self.compareHoldChanged.emit(True)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._apply_cursor()
            self.update()
        elif e.key() == Qt.Key_B and not e.isAutoRepeat():
            if self._hold_key:
                self._hold_key = False
                self.compareHoldChanged.emit(False)
        else:
            super().keyReleaseEvent(e)

    def focusOutEvent(self, e) -> None:
        """מקש ההחזקה (או Space) שנלחץ ואז הפוקוס עבר: אירוע השחרור לא יגיע, ולכן משחררים כאן."""
        if self._hold_key:
            self._hold_key = False
            self.compareHoldChanged.emit(False)
        if self._space:
            self._space = False
            self._apply_cursor()
            self.update()
        super().focusOutEvent(e)

    def resizeEvent(self, _) -> None:
        if self._fit_mode:
            self.fit()
        else:
            self.offset = self._clamped(self.scale, self.offset)
