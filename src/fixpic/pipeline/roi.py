"""חישוב ROI לפי ARCHITECTURE §4, §4ב ו-ADR-010/011.

הקלט למודל הוא תמיד ריבוע בגודל קבוע S (bucket: 256/384/512), כי OpenVINO דורש compile לכל צורה.
חלון ה-ROI נחתך סביב המסכה: אם ההקשר נכנס ב-S הוא נשלח בקנה מידה מקורי (חלון S×S, בלי Resize);
אחרת החלון גדל והוקטן ב-INTER_AREA ליחס אחיד (בלי עיוות). הרצועה שנותרה מרופדת עד S×S."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import EmptyMaskError


@dataclass(frozen=True)
class RoiPlan:
    x0: int
    y0: int
    x1: int           # בלעדי
    y1: int           # בלעדי
    scale: float      # 1.0 = קנה מידה מקורי; <1 = הקטנה
    size: int         # הצלע S של הקלט למודל
    in_w: int         # גודל החלון אחרי קנה המידה (לפני ריפוד עד S×S)
    in_h: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def downscaled(self) -> bool:
        return self.scale < 1.0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) בלעדי של הפיקסלים הלא-אפס; EmptyMaskError אם אין."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise EmptyMaskError("לא סומן אזור")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def plan_roi(mask: np.ndarray, size: int, margin_ratio: float = 0.25, min_margin: int = 16) -> RoiPlan:
    img_h, img_w = mask.shape[:2]
    bx0, by0, bx1, by1 = mask_bbox(mask)
    bw, bh = bx1 - bx0, by1 - by0
    margin = max(min_margin, math.ceil(margin_ratio * max(bw, bh)))
    needed = max(bw, bh) + 2 * margin
    win = max(needed, size)                 # צלע החלון בפיקסלי המקור
    scale = size / win                      # 1.0 כשההקשר נכנס ב-S
    ww, wh = min(win, img_w), min(win, img_h)
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    x0 = int(np.clip(cx - ww // 2, 0, img_w - ww))
    y0 = int(np.clip(cy - wh // 2, 0, img_h - wh))
    in_w = max(1, min(size, round(ww * scale)))
    in_h = max(1, min(size, round(wh * scale)))
    return RoiPlan(x0, y0, x0 + ww, y0 + wh, scale, size, in_w, in_h)
