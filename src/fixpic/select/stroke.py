"""משיכה → נקודות (AUTO-SELECT.md §7 שלב 4, ללא Qt): הופכת מסלול משיכת מברשת ל-`Prompts.points` בדגימה אחידה.

למה: GrabCut מצייר דיסק קטן סביב כל נקודה (רדיוס ~3% מצלע ה-ROI, לפחות 3 פיקסלים), ולכן נקודה בודדת אינה משנה דבר, ורק משיכה
**צפופה** מתקנת בחירה (נמדד, `tools/m4/grabcut_samples.py`). הממשק אוסף את אירועי העכבר ומעביר אותם לכאן.

חוזה:
- `path`: רצף (x, y) בפיקסלי התמונה המלאה, כפי שהגיעו מהעכבר: אפשר כפילויות, מרווחים לא אחידים וקפיצות; נקודה אחת = לחיצה.
- `brush_radius`: רדיוס המברשת בפיקסלי התמונה. הבחירה היא **השטח** שהמברשת מכסה (קפסולה סביב המסלול, כולל קצוות מעוגלים), לא רק קו המרכז.
- הדגימה אחידה על רשת (תא = `spacing`), ולכן: אין כפילויות כשהמשיכה חוזרת על עצמה, התוצאה דטרמיניסטית ואינה תלויה בקצב אירועי
  העכבר, והמסלול נדגם באותה צפיפות בכל מקום. `spacing` ברירת מחדל = `DEFAULT_SPACING` (כמו הרדיוס המינימלי של דיסק GrabCut), ולכן
  הדיסקים של GrabCut חופפים תמיד.
- תקרת `max_points`: אם יש יותר, ה-`spacing` גדל באופן אחיד (לא חותכים את סוף המשיכה).
- נקודות מחוץ לתמונה (`bounds=(w, h)`) נשמרות מחוץ לפלט (GrabCut מסרב לנקודה חיובית מחוץ לתמונה). משיכה כולה מחוץ לתמונה = `()`.
- קלט לא תקין (מסלול ריק, NaN/אינסוף, רדיוס או `spacing` לא חיוביים) = `SelectionError`.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import cv2
import numpy as np

from .types import Point, Prompts, SelectionError

DEFAULT_SPACING = 3.0            # פיקסלי תמונה; = `grabcut.MIN_HINT_RADIUS`
DEFAULT_MAX_POINTS = 2000
MAX_PATH_POINTS = 200_000
_MAX_CANVAS_CELLS = 4_000_000    # תקרת גודל הרשת הזמנית (ה-spacing גדל לפי הצורך)
_FX = 4                          # bits של נקודה קבועה ל-cv2 (מיקום תת-פיקסל)


def stroke_to_points(path: Sequence[tuple[float, float]], *, brush_radius: float, positive: bool = True,
                     bounds: tuple[int, int] | None = None, spacing: float = DEFAULT_SPACING,
                     max_points: int = DEFAULT_MAX_POINTS) -> tuple[Point, ...]:
    """מסלול משיכה → נקודות `Point` בדגימה אחידה על השטח שהמברשת כיסתה. ראו חוזה בראש הקובץ."""
    pts = _check_path(path)
    r = _check_positive(brush_radius, "brush_radius")
    sp = _check_positive(spacing, "spacing")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise SelectionError("max_points חייב להיות מספר שלם חיובי")
    max_points = int(max_points)

    lo = pts.min(axis=0) - r
    hi = pts.max(axis=0) + r
    if bounds is not None:
        try:
            bw, bh = int(bounds[0]), int(bounds[1])
        except (TypeError, ValueError, IndexError) as e:
            raise SelectionError("bounds חייב להיות (רוחב, גובה)") from e
        if bw < 1 or bh < 1:
            raise SelectionError("bounds חייב להיות חיובי")
        lo = np.maximum(lo, 0.0)
        hi = np.minimum(hi, (bw, bh))                    # בלעדי: נקודה חוקית היא x < bw
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            return ()
    x0, y0 = float(lo[0]), float(lo[1])
    ext_w, ext_h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    sp = max(sp, math.sqrt(ext_w * ext_h / _MAX_CANVAS_CELLS))

    for _ in range(24):
        xs, ys = _sample(pts, r, sp, x0, y0, ext_w, ext_h)
        if bounds is not None:
            keep = (xs >= 0) & (ys >= 0) & (xs < bounds[0]) & (ys < bounds[1])
            xs, ys = xs[keep], ys[keep]
        if len(xs) <= max_points:
            break
        sp *= max(1.02, math.sqrt(len(xs) / max_points))
    else:                                                # לא אמור לקרות; הגנה: חיתוך אחיד
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]

    if len(xs) == 0:                                     # משיכה זעירה שלא פגעה באף מרכז-תא: נקודת ההתחלה (אם בתוך התמונה)
        x, y = float(pts[0][0]), float(pts[0][1])
        if bounds is None or (0 <= x < bounds[0] and 0 <= y < bounds[1]):
            return (Point(x, y, bool(positive)),)
        return ()
    return tuple(Point(float(x), float(y), bool(positive)) for x, y in zip(xs, ys))


def with_stroke(prompts: Prompts, path: Sequence[tuple[float, float]], *, brush_radius: float, positive: bool = True,
                bounds: tuple[int, int] | None = None, spacing: float = DEFAULT_SPACING,
                max_points: int = DEFAULT_MAX_POINTS) -> Prompts:
    """`Prompts` חדש עם נקודות המשיכה מצורפות (שאר השדות, כולל `box`/`polygon`, ללא שינוי). המקור אינו משתנה."""
    new = stroke_to_points(path, brush_radius=brush_radius, positive=positive, bounds=bounds, spacing=spacing,
                           max_points=max_points)
    return dataclasses.replace(prompts, points=prompts.points + new)


# ---------------------------------------------------------------------------------------------------------------------
def _sample(pts: np.ndarray, r: float, sp: float, x0: float, y0: float, w: float, h: float):
    """מצייר את הקפסולה על רשת של תאים בגודל `sp` ומחזיר את מרכזי התאים הצבועים (בקואורדינטות תמונה)."""
    cw, ch = max(1, math.ceil(w / sp)), max(1, math.ceil(h / sp))
    canvas = np.zeros((ch, cw), np.uint8)
    one = 1 << _FX

    def fx(p) -> tuple[int, int]:                        # מרכז תא j = x0 + (j + 0.5)·sp  ←  קואורדינטת קנבס j
        return int(round(((p[0] - x0) / sp - 0.5) * one)), int(round(((p[1] - y0) / sp - 0.5) * one))

    rad = max(0, int(round(r / sp * one)))
    thick = max(1, int(round(2 * r / sp)))
    fpts = [fx(p) for p in pts]
    if len(fpts) > 1:
        cv2.polylines(canvas, [np.array(fpts, np.int32).reshape(-1, 1, 2)], False, 255, thick, cv2.LINE_8, _FX)
    for c in fpts:                                       # קצוות ופינות מעוגלים (וגם נקודה בודדת)
        cv2.circle(canvas, c, rad, 255, -1, cv2.LINE_8, _FX)
    iy, ix = np.nonzero(canvas)
    return x0 + (ix + 0.5) * sp, y0 + (iy + 0.5) * sp


def _check_path(path) -> np.ndarray:
    try:
        arr = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise SelectionError("מסלול משיכה לא תקין") from e
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        raise SelectionError("מסלול משיכה חייב להכיל לפחות נקודה אחת (x, y)")
    if len(arr) > MAX_PATH_POINTS:
        raise SelectionError("מסלול משיכה ארוך מדי")
    if not np.isfinite(arr).all():
        raise SelectionError("מסלול משיכה מכיל ערך לא סופי")
    keep = np.ones(len(arr), bool)                       # מסירים כפילויות עוקבות (אירועי עכבר חוזרים)
    keep[1:] = np.any(arr[1:] != arr[:-1], axis=1)
    return arr[keep]


def _check_positive(v, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise SelectionError(f"{name} חייב להיות מספר חיובי") from e
    if isinstance(v, bool) or not math.isfinite(f) or f <= 0:
        raise SelectionError(f"{name} חייב להיות מספר חיובי")
    return f
