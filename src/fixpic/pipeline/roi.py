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


def plan_roi(mask: np.ndarray, size: int, margin_ratio: float = 0.25, min_margin: int = 16,
             native_margin_ratio: float | None = None) -> RoiPlan:
    return plan_roi_bbox(mask_bbox(mask), mask.shape[:2], size, margin_ratio, min_margin, native_margin_ratio)


def plan_roi_bbox(box, img_hw, size: int, margin_ratio: float = 0.25, min_margin: int = 16,
                  native_margin_ratio: float | None = None) -> RoiPlan:
    """native_margin_ratio (§4ב): אם ההקשר המלא אינו נכנס ב-S אך הקשר מצומצם (עד יחס זה, לא פחות מ-min_margin) נכנס,
    החלון נשאר S×S בקנה מידה מקורי במקום הקטנה. None = כבוי (התנהגות M2.1–M2.7)."""
    img_h, img_w = img_hw
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    side = max(bw, bh)
    margin = max(min_margin, math.ceil(margin_ratio * side))
    needed = side + 2 * margin
    if needed > size and native_margin_ratio is not None:
        if side + 2 * max(min_margin, math.ceil(native_margin_ratio * side)) <= size:
            needed = size
    win = max(needed, size)                 # צלע החלון בפיקסלי המקור
    scale = size / win                      # 1.0 כשההקשר נכנס ב-S
    ww, wh = min(win, img_w), min(win, img_h)
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    x0 = int(np.clip(cx - ww // 2, 0, img_w - ww))
    y0 = int(np.clip(cy - wh // 2, 0, img_h - wh))
    in_w = max(1, min(size, round(ww * scale)))
    in_h = max(1, min(size, round(wh * scale)))
    return RoiPlan(x0, y0, x0 + ww, y0 + wh, scale, size, in_w, in_h)


def tile_plan(x0: int, y0: int, x1: int, y1: int, img_hw, size: int) -> RoiPlan:
    """חלון S×S בקנה מידה מקורי (מוצמד לתמונה) סביב ליבת אריח."""
    img_h, img_w = img_hw
    ww, wh = min(size, img_w), min(size, img_h)
    wx = int(np.clip((x0 + x1) // 2 - ww // 2, 0, img_w - ww))
    wy = int(np.clip((y0 + y1) // 2 - wh // 2, 0, img_h - wh))
    return RoiPlan(wx, wy, wx + ww, wy + wh, 1.0, size, ww, wh)


def tile_cores(box, size: int, overlap: int):
    """ליבות אריחים שמכסות את ה-box; כל ליבה + overlap מכל צד נכנסת בחלון S×S."""
    x0, y0, x1, y1 = box
    step = max(1, size - 2 * overlap)
    nx, ny = -(-(x1 - x0) // step), -(-(y1 - y0) // step)
    xs = np.linspace(x0, x1, nx + 1).round().astype(int)
    ys = np.linspace(y0, y1, ny + 1).round().astype(int)
    return [(int(xs[i]), int(ys[j]), int(xs[i + 1]), int(ys[j + 1])) for j in range(ny) for i in range(nx)]
