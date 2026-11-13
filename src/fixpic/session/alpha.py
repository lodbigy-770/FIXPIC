"""מדיניות Alpha בהסרה (ARCHITECTURE §5: המודל מקבל RGB בלבד, וה-Alpha נשמר בנפרד).

המדיניות: **בתוך האזור שהוסר, ה-Alpha מושלם מהסביבה** ולא נשמר ולא נכפה אטום. כך הסרת אובייקט שיושב על רקע
אטום משאירה אטום, הסרת אובייקט על רקע שקוף משאירה שקוף, והסרה בקצה (חצי שקוף) ממשיכה את הקצה. לשם כך
`cv2.inpaint` (Navier-Stokes, נבחר על Telea: על גרדיאנט Alpha נמדדה שגיאה ממוצעת 1.9 מול 5.4 ומקסימום 7 מול 26) על Alpha בלבד, במלבן המסכה עם שוליים, ועל עותק מוקטן עד `MAX_SIDE` כדי שהעלות תהיה חסומה
(נמדד: חור 1500² בתוך 2400² כ-1 שנייה ברזולוציה מלאה, כ-90ms בהקטנה ל-512). אין פיקסל ידוע בכלל (המסכה מכסה
את כל המלבן המורחב, כלומר את כל התמונה) = אטום.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 512
MARGIN = 16


def fill_alpha(alpha: np.ndarray, mask01: np.ndarray, box: tuple[int, int, int, int],
               sub_mask: np.ndarray) -> np.ndarray:
    """ערכי Alpha חדשים לפיקסלים שבמסכה, בסדר שורות ([N] uint8).

    alpha: Alpha הנוכחי [H,W]; mask01: המסכה האפקטיבית המלאה (uint8 0/1); box: המלבן הדק שלה; sub_mask: bool של
    המסכה בתוך ה-box. הקלט אינו משתנה."""
    h, w = alpha.shape
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = max(0, x0 - MARGIN), max(0, y0 - MARGIN), min(w, x1 + MARGIN), min(h, y1 + MARGIN)
    a = alpha[cy0:cy1, cx0:cx1]
    m = (mask01[cy0:cy1, cx0:cx1] > 0).astype(np.uint8)
    n = int(sub_mask.sum())
    if int(cv2.countNonZero(m)) == m.size:        # אין שום פיקסל ידוע
        return np.full(n, 255, np.uint8)
    ch, cw = a.shape
    s = min(1.0, MAX_SIDE / max(ch, cw))
    if s < 1.0:
        sw, sh = max(1, round(cw * s)), max(1, round(ch * s))
        # המסכה מוגדלת מעט כדי שערכי Alpha ישנים (של האובייקט) לא יזלגו לתוך הערכים שמתמלאים ממנה
        sm = cv2.dilate(cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
        sa = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
        if int(cv2.countNonZero(sm)) == sm.size:
            return np.full(n, 255, np.uint8)
        filled = cv2.resize(cv2.inpaint(sa, sm, 3, cv2.INPAINT_NS), (cw, ch), interpolation=cv2.INTER_LINEAR)
    else:
        filled = cv2.inpaint(np.ascontiguousarray(a), m, 3, cv2.INPAINT_NS)
    return filled[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0][sub_mask]
