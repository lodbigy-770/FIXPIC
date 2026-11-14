"""פעולות מסכה טהורות (M4, ללא UI): הרחבה/כיווץ, רכיבים מחוברים וניקוי רכיבים קטנים.

כל הפונקציות מקבלות מסכה uint8 [H,W] (0 = לשמירה, כל ערך אחר = להסרה) ומחזירות מסכה חדשה בערכים 0/255, בלי
לשנות את הקלט. הפעולות המורפולוגיות עובדות רק על המלבן שסביב המסכה (+ שוליים), כדי שמסכה קטנה על תמונה של 24MP
לא תעלה כמו תמונה שלמה.

גבול התמונה: הרחבה אינה "חוזרת" מהקצה; כיווץ אינו אוכל מהקצה (מסכה שנוגעת בקצה התמונה נשארת נוגעת בו, כי
המשתמש צייר עד הקצה ולא סימן שהקצה עצמו אינו חלק ממנה).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..errors import InvalidInputError

Rect = tuple[int, int, int, int]   # x0, y0, x1, y1 (x1/y1 בלעדיים)

MAX_MORPH_PX = 2000   # תקרת הגיוניות לרדיוס הרחבה/כיווץ
MAX_COMPONENTS = 100_000
MAX_POLY_POINTS = 30_000   # תקרת נקודות ללאסו (מספר ה-winding בשורה חסום במספר הצלעות, וחייב להיכנס ל-int16)
MAX_POLY_COORD = 1_000_000
_POLY_CHUNK_ROWS = 1 << 21   # חיתוך החישוב: לכל היותר כ-2M חציות בזיכרון בבת אחת


@dataclass(frozen=True)
class Component:
    id: int          # מספור יציב לפי מיקום (למעלה-שמאל קודם); תקף רק למסכה שממנה חושב
    bbox: Rect
    area: int        # מספר פיקסלים
    centroid: tuple[float, float]   # x, y


def _check(mask) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise InvalidInputError("נדרשת מסכה [H,W] לא ריקה")
    return mask


def _check_px(px) -> int:
    if isinstance(px, bool) or not isinstance(px, (int, np.integer)) or not 1 <= px <= MAX_MORPH_PX:
        raise InvalidInputError(f"רדיוס חייב להיות מספר שלם בין 1 ל-{MAX_MORPH_PX}")
    return int(px)


def _bin(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def bbox_of(mask: np.ndarray) -> Rect | None:
    x, y, w, h = cv2.boundingRect(_bin(_check(mask)))
    return None if w == 0 or h == 0 else (x, y, x + w, y + h)


def morph(mask: np.ndarray, px: int, *, grow: bool) -> tuple[np.ndarray, Rect | None]:
    """הרחבה (`grow=True`) או כיווץ ב-`px` פיקסלים (גרעין אליפטי, כמו ההרחבה האוטומטית של ה-Pipeline).
    מחזיר (מסכה חדשה 0/255, המלבן ששונה או None אם אין שינוי/המסכה ריקה)."""
    _check(mask)
    r = _check_px(px)
    b = _bin(mask)
    box = bbox_of(mask)
    out = b * 255
    if box is None:
        return out, None
    h, w = b.shape
    x0, y0, x1, y1 = box
    pad = r + 1
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    crop = np.ascontiguousarray(b[cy0:cy1, cx0:cx1])
    res = cv2.dilate(crop, k) if grow else cv2.erode(crop, k)   # ברירת מחדל של cv2: הקצה אינו משפיע
    out[cy0:cy1, cx0:cx1] = res * 255
    changed = (res != crop)
    if not changed.any():
        return out, None
    ys, xs = np.where(changed)
    return out, (cx0 + int(xs.min()), cy0 + int(ys.min()), cx0 + int(xs.max()) + 1, cy0 + int(ys.max()) + 1)


def components(mask: np.ndarray, *, min_area: int = 1, connectivity: int = 8) -> list[Component]:
    """הרכיבים המחוברים של המסכה, ממוינים לפי מיקום (שורה עליונה, אחר כך שמאל). `min_area` מסנן רכיבים קטנים.
    connectivity: 8 (ברירת מחדל, כמו קיבוץ המסכות של ה-Pipeline) או 4."""
    _check(mask)
    if connectivity not in (4, 8) or isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 1:
        raise InvalidInputError("connectivity חייב להיות 4 או 8, ו-min_area מספר שלם חיובי")
    n, _, stats, cent = cv2.connectedComponentsWithStats(_bin(mask), connectivity=connectivity)
    if n - 1 > MAX_COMPONENTS:
        raise InvalidInputError(f"יותר מדי רכיבים במסכה ({n - 1})")
    items = [(int(stats[i, 1]), int(stats[i, 0]), i) for i in range(1, n) if stats[i, 4] >= min_area]
    items.sort()
    out = []
    for new_id, (_, _, i) in enumerate(items, 1):
        x, y, w, h, a = (int(v) for v in stats[i])
        out.append(Component(new_id, (x, y, x + w, y + h), a, (float(cent[i][0]), float(cent[i][1]))))
    return out


def component_at(mask: np.ndarray, x: float, y: float, *, connectivity: int = 8) -> np.ndarray | None:
    """מסכה (0/255, בגודל הקלט) של הרכיב שמכיל את הנקודה, או None אם הנקודה מחוץ לתמונה או על רקע.
    עובד על מלבן הרכיב בלבד (floodFill), לא על התמונה כולה."""
    _check(mask)
    if connectivity not in (4, 8):
        raise InvalidInputError("connectivity חייב להיות 4 או 8")
    try:
        xi, yi = int(round(float(x))), int(round(float(y)))
    except (TypeError, ValueError, OverflowError) as e:
        raise InvalidInputError("נקודה לא תקינה") from e
    h, w = mask.shape
    if not (0 <= xi < w and 0 <= yi < h) or mask[yi, xi] == 0:
        return None
    b = _bin(mask)
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(b, ff, (xi, yi), 2, flags=connectivity | cv2.FLOODFILL_FIXED_RANGE)
    return (b == 2).astype(np.uint8) * 255


def remove_small(mask: np.ndarray, min_area: int, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """מוחק רכיבים קטנים מ-`min_area` פיקסלים (ניקוי כתמים אחרי ציור). מחזיר (מסכה חדשה, כמה רכיבים נמחקו)."""
    _check(mask)
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 1:
        raise InvalidInputError("min_area חייב להיות מספר שלם חיובי")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_bin(mask), connectivity=connectivity)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (stats[1:, 4] >= min_area)
    removed = int((stats[1:, 4] < min_area).sum()) if n > 1 else 0
    return keep[labels] * 255, removed


# ------------------------------------------------------------------ לאסו (מצולע חופשי)
def _check_polygon(points) -> np.ndarray:
    try:
        a = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise InvalidInputError("נקודות לאסו לא תקינות") from e
    if a.ndim != 2 or a.shape[1] != 2:
        raise InvalidInputError("נקודות לאסו חייבות להיות רשימה של (x, y)")
    if len(a) < 3:
        raise InvalidInputError("לאסו דורש לפחות 3 נקודות")
    if len(a) > MAX_POLY_POINTS:
        raise InvalidInputError(f"יותר מדי נקודות בלאסו ({len(a)}); יש לפשט את המסלול לפני החישוב")
    if not np.isfinite(a).all() or np.abs(a).max() > MAX_POLY_COORD:
        raise InvalidInputError("קואורדינטת לאסו לא סופית או גדולה מדי")
    return np.rint(a).astype(np.int64)


def polygon_region(points, shape: tuple[int, int]) -> tuple[np.ndarray, Rect] | None:
    """ממלא את המצולע הסגור שנוצר מהנקודות (המסלול נסגר אוטומטית מהנקודה האחרונה לראשונה), לכלי הלאסו.

    מחזיר (חלון 0/255, מלבנו בתמונה) כשהחלון הוא חיתוך המצולע עם תמונה בגודל `shape` (h, w), או None אם אין
    כלל מה למלא (כולו מחוץ לתמונה, או ששטחו אפס כי הנקודות על קו אחד). העבודה והזיכרון לפי החלון ולא לפי התמונה.

    כלל המילוי: winding שונה מאפס, כלומר איחוד כל הלולאות גם כשהמסלול חוצה את עצמו או מקיף פעמיים (בניגוד
    ל-`cv2.fillPoly`, שמשאיר חור באזור שהמסלול הקיף פעמיים). קואורדינטות עגולות לפיקסל (מרכז פיקסל = מספר שלם,
    כמו ב-`MaskLayer.paint`); הפיקסלים שעל הקו עצמו נכללים, כך שמלבן (10,10)–(50,40) ממלא את 10..50 ו-10..40 כולל.
    הפנים מחושבים בחשבון שלם ואינם תלויים בחיתוך; קו השפה (1px) צויר ע"י cv2, וכשהמצולע חורג מהתמונה פיקסל שפה
    בודד על קו אלכסוני עשוי להשתנות לעומת מצולע שלם שנחתך אחר כך.
    """
    h, w = int(shape[0]), int(shape[1])
    if h < 1 or w < 1:
        raise InvalidInputError("גודל תמונה לא תקין")
    pts = _check_polygon(points)
    x, y = pts[:, 0], pts[:, 1]
    cx0, cy0 = max(0, int(x.min())), max(0, int(y.min()))
    cx1, cy1 = min(w, int(x.max()) + 1), min(h, int(y.max()) + 1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    cw, ch = cx1 - cx0, cy1 - cy0

    # צלעות סגורות; אופקיות לא משפיעות על ה-winding. כל צלע תורמת לשורות [ylo, yhi) (חצי פתוח: אין ספירה כפולה בקודקוד)
    ax, ay, bx, by = x, y, np.roll(x, -1), np.roll(y, -1)
    r0, r1 = np.maximum(np.minimum(ay, by), cy0), np.minimum(np.maximum(ay, by), cy1)
    ok = (ay != by) & (r1 > r0)
    ax, ay, bx, by, r0, r1 = ax[ok], ay[ok], bx[ok], by[ok], r0[ok], r1[ok]
    sign = np.where(by > ay, 1, -1).astype(np.int16)
    cnt = r1 - r0

    delta = np.zeros(ch * (cw + 1), np.int16)   # עמודה אחרונה = "מימין לחלון": לא משפיעה
    i, n = 0, len(cnt)
    while i < n:
        j = i + 1
        tot = int(cnt[i])
        while j < n and tot + int(cnt[j]) <= _POLY_CHUNK_ROWS:
            tot += int(cnt[j])
            j += 1
        c = cnt[i:j]
        idx = np.repeat(np.arange(i, j), c)
        rows = r0[idx] + (np.arange(tot) - np.repeat(np.cumsum(c) - c, c))
        # נקודת החיתוך עם השורה, בחשבון שלם בדיוק: ceil(ax + (row-ay)*(bx-ax)/(by-ay)); הפיקסלים מ-ceil ואילך "בפנים"
        dy = by[idx] - ay[idx]
        num = ax[idx] * dy + (rows - ay[idx]) * (bx[idx] - ax[idx])
        sgn = np.where(dy < 0, -1, 1)
        col = -((-num * sgn) // (dy * sgn))
        col = np.clip(col - cx0, 0, cw)
        np.add.at(delta, (rows - cy0) * (cw + 1) + col, sign[idx])
        i = j

    wind = np.cumsum(delta.reshape(ch, cw + 1), axis=1, dtype=np.int16)[:, :cw]
    inside = wind != 0
    if not inside.any():
        return None
    out = inside.astype(np.uint8) * 255
    outline = (pts - np.array([cx0, cy0], np.int64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [outline], True, 255, 1, cv2.LINE_8)
    return out, (cx0, cy0, cx1, cy1)
