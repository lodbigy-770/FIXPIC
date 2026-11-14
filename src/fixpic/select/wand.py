"""שרביט קסם (רמה 0, AUTO-SELECT.md §3): `floodFill` לפי דמיון צבע לנקודת ההתחלה, על "תמונת ניתוח" מוקטנת.

חוזים:
- `prepare` מכין עותק פרטי של התמונה (`floodFill` דורש מערך כתיב, אך עם `MASK_ONLY` אינו משנה אותו; נבדק), מוקטן ב-INTER_AREA ל-`MAX_ANALYSIS_PX` פיקסלים לכל היותר (אין הקטנה
  לתמונה קטנה יותר). העלות זניחה, ו-Context תלוי בצבעי הפיקסלים: **אחרי Commit חייבים Context חדש** (`stale_ok=False`).
- `propose`: כל נקודה חיובית היא זרע; האיחוד של אזורי ה-floodFill שלהן (טווח קבוע מול צבע הזרע, כמו שרביט קלאסי,
  לא מול השכן) פחות האיחוד של אזורי הנקודות השליליות. חיבוריות 8. `box` חותך את אזור החיפוש (הזרעים החיוביים
  חייבים להיות בתוכו). `polygon` אינו נתמך (`SelectionError`). בלי נקודה חיובית = `SelectionError`; תוצאה ריקה
  (שלילית כיסתה הכול) = None.
- המסכה מוחזרת ברזולוציה מלאה: חלון הניתוח מוגדל ב-INTER_LINEAR וסף 127, כך שהמלבן והחלון בגודל התמונה המלאה.
- `tolerance` 0..255 לכל ערוץ; ברירת מחדל `DEFAULT_TOLERANCE` (זמני, **לא כוונן על תמונות אמיתיות**).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..errors import InvalidInputError
from ..models.base import CancelToken
from .types import Context, Point, Prompts, Proposal, SelectionError

MAX_ANALYSIS_PX = 1_500_000
DEFAULT_TOLERANCE = 20          # ערך זמני
_FLAGS = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)


class MagicWandProvider:
    name = "magic-wand"
    heavy = False
    stale_ok = False

    def __init__(self, tolerance: int = DEFAULT_TOLERANCE, max_analysis_px: int = MAX_ANALYSIS_PX) -> None:
        _check_tolerance(tolerance)
        if max_analysis_px < 64:
            raise ValueError("max_analysis_px קטן מדי")
        self.tolerance = tolerance
        self.max_analysis_px = max_analysis_px

    # ------------------------------------------------------------------ prepare
    def prepare(self, rgb: np.ndarray, cancel: CancelToken | None = None) -> Context:
        if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3):
            raise InvalidInputError("התמונה חייבת להיות uint8 [H,W,3]")
        h, w = rgb.shape[:2]
        if h < 1 or w < 1:
            raise InvalidInputError("גודל תמונה לא תקין")
        if cancel:
            cancel.raise_if_cancelled()
        if h * w > self.max_analysis_px:
            s = math.sqrt(self.max_analysis_px / (h * w))
            ah, aw = max(1, round(h * s)), max(1, round(w * s))
            small = cv2.resize(rgb, (aw, ah), interpolation=cv2.INTER_AREA)
        else:
            small = np.array(rgb, copy=True)
        small = np.ascontiguousarray(small)
        return Context(self.name, (h, w), small, int(small.nbytes))

    # ------------------------------------------------------------------ propose
    def propose(self, ctx: Context, prompts: Prompts, cancel: CancelToken | None = None) -> Proposal | None:
        if ctx.provider != self.name:
            raise SelectionError("Context שייך ל-Provider אחר")
        if prompts.polygon is not None:
            raise SelectionError("השרביט אינו תומך בלאסו (polygon)")
        pos, neg = prompts.positives, prompts.negatives
        if not pos:
            raise SelectionError("חסרה נקודה חיובית")
        tol = self.tolerance if prompts.tolerance is None else _check_tolerance(prompts.tolerance)
        small: np.ndarray = ctx.data
        h, w = ctx.shape
        ah, aw = small.shape[:2]
        sx, sy = w / aw, h / ah

        if prompts.box is not None:                      # אזור חיפוש באנליזה, מעוגל כלפי חוץ
            bx0, by0, bx1, by1 = _check_box(prompts.box, w, h)
            ax0, ay0 = int(bx0 / sx), int(by0 / sy)
            ax1, ay1 = min(aw, math.ceil(bx1 / sx)), min(ah, math.ceil(by1 / sy))
        else:
            ax0, ay0, ax1, ay1 = 0, 0, aw, ah
        view = small[ay0:ay1, ax0:ax1]
        vh, vw = view.shape[:2]

        def region(points: tuple[Point, ...], strict: bool) -> np.ndarray:
            acc = np.zeros((vh, vw), np.uint8)
            for p in points:
                _check_point(p, w, h)
                px, py = min(aw - 1, int(p.x / sx)) - ax0, min(ah - 1, int(p.y / sy)) - ay0
                if not (0 <= px < vw and 0 <= py < vh):
                    if strict:
                        raise SelectionError("נקודה חיובית מחוץ למלבן החיפוש")
                    continue
                if cancel:
                    cancel.raise_if_cancelled()
                m = np.zeros((vh + 2, vw + 2), np.uint8)
                cv2.floodFill(view, m, (px, py), 0, (tol, tol, tol), (tol, tol, tol), _FLAGS)
                acc |= m[1:-1, 1:-1]
            return acc

        sel = region(pos, True)
        if neg:
            sel[region(neg, False) > 0] = 0
        if cancel:
            cancel.raise_if_cancelled()
        x, y, bw, bh = cv2.boundingRect(sel)
        if bw == 0 or bh == 0:
            return None
        crop = sel[y:y + bh, x:x + bw]
        # מלבן ברזולוציה מלאה, בתוך התמונה ובגודל לפחות פיקסל
        fx0, fy0 = min(round((ax0 + x) * sx), w - 1), min(round((ay0 + y) * sy), h - 1)
        fx1 = min(max(round((ax0 + x + bw) * sx), fx0 + 1), w)
        fy1 = min(max(round((ay0 + y + bh) * sy), fy0 + 1), h)
        if (fx1 - fx0, fy1 - fy0) != (bw, bh):
            crop = cv2.resize(crop, (fx1 - fx0, fy1 - fy0), interpolation=cv2.INTER_LINEAR)
        window = np.where(crop > 127, 255, 0).astype(np.uint8)
        if not window.any():
            return None
        return Proposal(window, (fx0, fy0, fx1, fy1))


def _check_tolerance(t) -> int:
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 0 <= int(t) <= 255:
        raise SelectionError("tolerance חייב להיות מספר שלם בין 0 ל-255")
    return int(t)


def _check_point(p: Point, w: int, h: int) -> None:
    try:
        ok = math.isfinite(p.x) and math.isfinite(p.y)
    except TypeError as e:
        raise SelectionError("נקודה לא תקינה") from e
    if not ok:
        raise SelectionError("נקודה לא תקינה")
    if not (0 <= p.x < w and 0 <= p.y < h):
        raise SelectionError("נקודה מחוץ לתמונה")


def _check_box(box, w: int, h: int):
    try:
        x0, y0, x1, y1 = (int(v) for v in box)
    except (TypeError, ValueError) as e:
        raise SelectionError("מלבן לא תקין") from e
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise SelectionError("המלבן ריק או מחוץ לתמונה")
    return x0, y0, x1, y1
