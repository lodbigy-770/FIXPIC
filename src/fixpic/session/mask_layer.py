"""MaskLayer (ARCHITECTURE §5): שכבת מסכה נפרדת מעל התמונה, uint8 [H,W] בגודל התמונה. 255 = להסרה, 0 = לשמירה.

השכבה אינה משנה את התמונה ואינה מכירה את ה-Pipeline. ציור הוא משיכות עגולות בינאריות (קו עם קצוות עגולים
בקוטר המברשת, בפיקסלי התמונה); ה-Feathering וההרחבה האוטומטית נעשים ב-Pipeline.

לא בטוחה לשימוש מכמה threads: ה-UI קורא לה מה-thread שלו בלבד.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..errors import InvalidInputError

Rect = tuple[int, int, int, int]   # x0, y0, x1, y1 (x1/y1 בלעדיים)

MAX_COORD = 1_000_000   # תקרת הגיוניות לקואורדינטה (cv2 עובד ב-int32)
MAX_DIAMETER = 100_000


class MaskLayer:
    def __init__(self, height: int, width: int) -> None:
        if height < 1 or width < 1:
            raise InvalidInputError("גודל שכבת מסכה לא תקין")
        self._m = np.zeros((height, width), np.uint8)
        self.version = 0              # עולה בכל שינוי בפועל (ל-UI: ציור מחדש)
        self._brush_px: float | None = None
        self._count_cache: tuple[int, int] | None = None   # (version, countNonZero)

    # ------------------------------------------------------------------ מצב
    @property
    def shape(self) -> tuple[int, int]:
        return self._m.shape

    @property
    def array(self) -> np.ndarray:
        """תצוגת קריאה בלבד של המסכה (לא להחזיק מעבר לשינוי הבא; להעתקה: `snapshot`)."""
        v = self._m.view()
        v.setflags(write=False)
        return v

    @property
    def pixel_count(self) -> int:
        if self._count_cache is None or self._count_cache[0] != self.version:
            self._count_cache = (self.version, int(cv2.countNonZero(self._m)))
        return self._count_cache[1]

    @property
    def empty(self) -> bool:
        return self.pixel_count == 0

    @property
    def brush_px(self) -> float | None:
        """הקוטר הגדול ביותר שצויר מאז הניקוי האחרון (פיקסלי תמונה), או None. מזין את `brush_px` של ה-Job."""
        return self._brush_px

    def bbox(self) -> Rect | None:
        """המלבן הדק סביב המסכה, או None אם ריקה."""
        x, y, w, h = cv2.boundingRect(self._m)
        return None if w == 0 or h == 0 else (x, y, x + w, y + h)

    def snapshot(self) -> np.ndarray:
        """עותק עצמאי של המסכה (מה ש-Job מקבל)."""
        return self._m.copy()

    # ---------------------------------------------------------------- ציור
    def paint(self, points: Sequence[tuple[float, float]], diameter: float, *, erase: bool = False) -> Rect | None:
        """משיכה בקוטר `diameter` לאורך הנקודות (נקודה אחת = עיגול). `erase=True` מוחק. קואורדינטות בפיקסלי
        התמונה, ויכולות לחרוג מהתמונה (נחתכות). מחזיר את המלבן שנגע בו (לציור מחדש חלקי) או None אם כולו מחוץ."""
        pts = _check_points(points)
        d = _check_diameter(diameter)
        t = max(1, int(round(d)))
        h, w = self._m.shape
        pad = t // 2 + 1
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        rect = (max(0, min(xs) - pad), max(0, min(ys) - pad), min(w, max(xs) + pad + 1), min(h, max(ys) + pad + 1))
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            return None
        val = 0 if erase else 255
        if len(pts) == 1:
            cv2.line(self._m, pts[0], pts[0], val, t, cv2.LINE_8)
        else:
            cv2.polylines(self._m, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, val, t, cv2.LINE_8)
        if not erase:
            self._brush_px = d if self._brush_px is None else max(self._brush_px, d)
        self._touch()
        return rect

    def erase(self, points: Sequence[tuple[float, float]], diameter: float) -> Rect | None:
        return self.paint(points, diameter, erase=True)

    def add(self, mask: np.ndarray) -> None:
        """איחוד (OR) מסכה שלמה בגודל השכבה לתוכה (שחזור מסכה אחרי כשל, או מסכה שנוצרה חיצונית)."""
        self._check_mask(mask)
        np.bitwise_or(self._m, np.where(mask > 0, 255, 0).astype(np.uint8), out=self._m)
        self._touch()

    def clear(self) -> None:
        """מנקה הכול (גם את `brush_px`)."""
        if self._m.any():
            self._m.fill(0)
            self._touch()
        self._brush_px = None

    def take(self) -> np.ndarray:
        """snapshot + clear: מוציא את המסכה מהשכבה."""
        s = self.snapshot()
        self.clear()
        return s

    # ---------------------------------------------------------------- פנימי
    def _touch(self) -> None:
        self.version += 1

    def _check_mask(self, mask: np.ndarray) -> None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != self._m.shape:
            raise InvalidInputError("המסכה חייבת להיות [H,W] בגודל השכבה")


def _check_points(points: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError) as e:
            raise InvalidInputError("נקודת משיכה לא תקינה") from e
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > MAX_COORD or abs(y) > MAX_COORD:
            raise InvalidInputError("קואורדינטת משיכה לא סופית או גדולה מדי")
        out.append((int(round(x)), int(round(y))))
    if not out:
        raise InvalidInputError("משיכה ללא נקודות")
    return out


def _check_diameter(d: float) -> float:
    try:
        d = float(d)
    except (TypeError, ValueError) as e:
        raise InvalidInputError("קוטר מברשת לא תקין") from e
    if not math.isfinite(d) or d <= 0 or d > MAX_DIAMETER:
        raise InvalidInputError("קוטר מברשת חייב להיות חיובי וסופי")
    return d
