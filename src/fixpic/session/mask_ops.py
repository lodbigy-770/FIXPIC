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
