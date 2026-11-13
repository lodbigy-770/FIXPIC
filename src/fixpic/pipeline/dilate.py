"""הרחבת מסכה אוטומטית (ADR-008, ARCHITECTURE §4 שלב 2): גודל ההרחבה יחסי לגודל המברשת, וניתן לכיבוי.

גודל המברשת (קוטר בפיקסלי התמונה) מגיע מה-UI. כשאינו ידוע, מוערך מהמסכה: למשיכה באורך L ועובי w, שטח A=w·L
והיקף (פיקסלי שפה) B≈2·L, ולכן w≈2A/B. ההערכה מיועדת למשיכות; לשטח מלא גדול היא נותנת ערך קטן מהקוטר, וממילא
ההרחבה חסומה מלמעלה. כל הערכים (יחס, מינימום, מקסימום) זמניים עד בדיקת איכות על Big LaMa האמיתי."""
from __future__ import annotations

import cv2
import numpy as np

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def estimate_brush_px(m: np.ndarray) -> float:
    """קוטר משיכה מוערך (פיקסלים) ממסכה uint8 0/1. 0 אם המסכה ריקה. זיכרון: עותק אחד של ה-bbox."""
    x, y, w, h = cv2.boundingRect(m)
    if w == 0 or h == 0:
        return 0.0
    crop = m[y:y + h, x:x + w]
    area = cv2.countNonZero(crop)
    edge = area - cv2.countNonZero(cv2.erode(crop, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    return 2.0 * area / max(edge, 1)


def dilate_radius(fixed_px: int | None, brush_px: float, ratio: float, lo: int, hi: int) -> int:
    """fixed_px אינו None: ערך קבוע (0 = כבוי). אחרת: ratio·קוטר המברשת, חסום בין lo ל-hi."""
    if fixed_px is not None:
        return max(0, int(fixed_px))
    return int(min(hi, max(lo, round(ratio * brush_px))))
