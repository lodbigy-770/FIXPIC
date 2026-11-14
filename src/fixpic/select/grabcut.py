"""GrabCut (רמה 0, AUTO-SELECT.md §3): מפריד אובייקט מרקע ממלבן גס או מלאסו גס, לפי מודל צבע, על ROI מוקטן.

חוזים:
- `prepare` מכין עותק פרטי של התמונה, מוקטן ב-INTER_AREA ל-`max_analysis_px` לכל היותר (אין הקטנה לתמונה קטנה יותר).
  Context תלוי בצבעים, ולכן **אחרי Commit חייבים Context חדש** (`stale_ok=False`). העלות זניחה.
- `propose` דורש `box` או `polygon` (לא שניהם). ה-ROI הוא תיבת הקלט פלוס שוליים (`margin`, שמספקים לו דוגמאות רקע),
  והוא **מוקטן ב-INTER_AREA ל-`max_roi_px` לכל היותר** (ברירת מחדל 0.27MP; ראו נתוני הזמן ב-AUTO-SELECT.md §3):
  GrabCut חד-ליבתי והזמן שלו גדל מהר עם השטח. מחוץ לקלט = רקע וודאי; בפנים = חזית אפשרית.
- נקודות: חיובית = חזית וודאית, שלילית = רקע וודאי (דיסק סביבה ברדיוס `hint_frac` מצלע ה-ROI הקצרה, לפחות 3 פיקסלים).
  נקודה חיובית חייבת להיות בתוך ה-ROI. `tolerance` אינו נתמך (`SelectionError`).
  **נמדד על `samples/` (מלבן רפוי שנוגע בקצה התמונה כולל את הספסל): נקודה אחת עם דיסק קטן אינה משנה דבר, ו-3 נקודות עם רדיוס
  של כ-3% מנקות אותו.** לכן ממשק ה-Scribble (שלב 4) צריך לשלוח משיכה כנקודות צפופות, לא לחיצה בודדת.
- אחרי ההרצה: משאירים כל רכיב ששטחו לפחות `KEEP_RATIO` (20%) משל הגדול ביותר, וכל רכיב שמכיל נקודה חיובית; כתמים קטנים
  נזרקים. החורים נשארים. (מדידה על `samples/`: מלבן סביב פס של 10 שלטים נתן 10 רכיבים כמעט שווים; \"הגדול בלבד\" זרק 9.)
- המסכה מוגדלת לרזולוציה מלאה: חלון הניתוח מוגדל ב-INTER_LINEAR וסף 127 (כמו בשרביט). שיפור קצה לפי התמונה: שאלה פתוחה.
- דטרמיניסטי: אותו קלט נותן אותה הצעה (זרע ה-RNG של OpenCV נקבע לפני כל הרצה, ב-thread של הקריאה).
- `heavy=True`: ה-Service ימתין כש-Inpaint פעיל. ביטול נבדק בין האיטרציות.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..errors import InvalidInputError
from ..models.base import CancelToken
from .types import Context, Prompts, Proposal, SelectionError
from .wand import _check_box, _check_point

MAX_ANALYSIS_PX = 4_000_000
MAX_ROI_PX = 270_000            # ~0.26MP: כ-0.3 שניות (ליבה אחת, סינתטי), AUTO-SELECT.md §3
ITERATIONS = 3
MARGIN = 0.10                   # שוליים סביב הקלט, חלק מגודלו
MIN_MARGIN_PX = 8
MAX_POLYGON_POINTS = 30_000     # כמו `MaskLayer.fill_polygon`
MIN_SIDE_PX = 8                 # מתחת לכך (בקנה מידה של הניתוח) אין מספיק פיקסלים למודל צבע
KEEP_RATIO = 0.2                # רכיב נשמר אם שטחו לפחות כך מהרכיב הגדול ביותר
HINT_FRAC = 0.03                # רדיוס דיסק נקודה, חלק מצלע ה-ROI הקצרה (ערך זמני; ראו הערה בראש הקובץ)
MIN_HINT_RADIUS = 3
_BGD, _FGD, _PR_BGD, _PR_FGD = cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD


class GrabCutProvider:
    name = "grabcut"
    heavy = True
    stale_ok = False

    def __init__(self, iterations: int = ITERATIONS, max_roi_px: int = MAX_ROI_PX,
                 max_analysis_px: int = MAX_ANALYSIS_PX, margin: float = MARGIN, hint_frac: float = HINT_FRAC) -> None:
        if not 1 <= iterations <= 20:
            raise ValueError("iterations חייב להיות בין 1 ל-20")
        if max_roi_px < 1024 or max_analysis_px < 1024:
            raise ValueError("גודל ניתוח קטן מדי")
        if not 0.0 <= margin <= 1.0:
            raise ValueError("margin חייב להיות בין 0 ל-1")
        if not 0.0 <= hint_frac <= 0.5:
            raise ValueError("hint_frac חייב להיות בין 0 ל-0.5")
        self.iterations, self.max_roi_px = iterations, max_roi_px
        self.max_analysis_px, self.margin, self.hint_frac = max_analysis_px, margin, hint_frac

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
            small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        else:
            small = np.array(rgb, copy=True)
        small = np.ascontiguousarray(small)
        return Context(self.name, (h, w), small, int(small.nbytes))

    # ------------------------------------------------------------------ propose
    def propose(self, ctx: Context, prompts: Prompts, cancel: CancelToken | None = None) -> Proposal | None:
        if ctx.provider != self.name:
            raise SelectionError("Context שייך ל-Provider אחר")
        if prompts.tolerance is not None:
            raise SelectionError("GrabCut אינו תומך ב-tolerance")
        if (prompts.box is None) == (prompts.polygon is None):
            raise SelectionError("GrabCut דורש מלבן או לאסו (בדיוק אחד מהם)")
        small: np.ndarray = ctx.data
        h, w = ctx.shape
        ah, aw = small.shape[:2]
        sx, sy = w / aw, h / ah

        # --- קלט בקואורדינטות הניתוח: מלבן הקלט, ומצולע (אם יש)
        poly_a: np.ndarray | None = None
        if prompts.box is not None:
            bx0, by0, bx1, by1 = _check_box(prompts.box, w, h)
            in_x0, in_y0 = int(bx0 / sx), int(by0 / sy)
            in_x1, in_y1 = min(aw, math.ceil(bx1 / sx)), min(ah, math.ceil(by1 / sy))
        else:
            poly_a = _polygon_to_analysis(prompts.polygon, w, h, sx, sy)
            px, py, pw, ph = cv2.boundingRect(poly_a)
            in_x0, in_y0, in_x1, in_y1 = max(0, px), max(0, py), min(aw, px + pw + 1), min(ah, py + ph + 1)
        if in_x1 - in_x0 < MIN_SIDE_PX or in_y1 - in_y0 < MIN_SIDE_PX:
            raise SelectionError("הקלט קטן מדי ל-GrabCut")

        # --- ROI = קלט + שוליים, בתוך התמונה
        mx = max(MIN_MARGIN_PX, round((in_x1 - in_x0) * self.margin))
        my = max(MIN_MARGIN_PX, round((in_y1 - in_y0) * self.margin))
        rx0, ry0, rx1, ry1 = max(0, in_x0 - mx), max(0, in_y0 - my), min(aw, in_x1 + mx), min(ah, in_y1 + my)
        rh, rw = ry1 - ry0, rx1 - rx0

        # --- הקטנת ה-ROI לתקציב (INTER_AREA לתמונה, NEAREST למסכה)
        k = min(1.0, math.sqrt(self.max_roi_px / (rh * rw)))
        gh, gw = max(1, round(rh * k)), max(1, round(rw * k))
        roi = small[ry0:ry1, rx0:rx1]
        img = cv2.resize(roi, (gw, gh), interpolation=cv2.INTER_AREA) if (gh, gw) != (rh, rw) else np.array(roi)
        img = np.ascontiguousarray(img)
        fx, fy = gw / rw, gh / rh

        def to_roi(x: float, y: float) -> tuple[int, int]:      # נקודה בקואורדינטות התמונה המלאה → פיקסל ב-ROI המוקטן
            return int((x / sx - rx0) * fx), int((y / sy - ry0) * fy)

        mask = np.full((gh, gw), _BGD, np.uint8)
        if poly_a is not None:
            shifted = np.round((poly_a.astype(np.float64) - (rx0, ry0)) * (fx, fy)).astype(np.int32)
            inside = np.zeros((gh, gw), np.uint8)
            cv2.fillPoly(inside, [shifted], 255)
            mask[inside > 0] = _PR_FGD
        else:
            ix0, iy0 = round((in_x0 - rx0) * fx), round((in_y0 - ry0) * fy)
            ix1, iy1 = round((in_x1 - rx0) * fx), round((in_y1 - ry0) * fy)
            mask[iy0:iy1, ix0:ix1] = _PR_FGD

        radius = max(MIN_HINT_RADIUS, round(self.hint_frac * min(gh, gw)))
        pos_pts: list[tuple[int, int]] = []
        for p in prompts.points:
            _check_point(p, w, h)
            qx, qy = to_roi(p.x, p.y)
            if not (0 <= qx < gw and 0 <= qy < gh):
                if p.positive:
                    raise SelectionError("נקודה חיובית מחוץ לאזור הניתוח")
                continue
            cv2.circle(mask, (qx, qy), radius, int(_FGD if p.positive else _BGD), -1)
            if p.positive:
                pos_pts.append((qx, qy))

        n_fg = int(np.count_nonzero((mask == _FGD) | (mask == _PR_FGD)))
        n_bg = int(np.count_nonzero((mask == _BGD) | (mask == _PR_BGD)))
        if n_fg < 25:
            raise SelectionError("הקלט קטן מדי ל-GrabCut")
        if n_bg < 25:
            raise SelectionError("אין מספיק רקע סביב הקלט (המלבן מכסה את כל התמונה)")

        # --- GrabCut, איטרציה-איטרציה כדי שאפשר יהיה לבטל
        if cancel:
            cancel.raise_if_cancelled()
        cv2.setRNGSeed(0)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            for i in range(self.iterations):
                cv2.grabCut(img, mask, None, bgd, fgd, 1, cv2.GC_INIT_WITH_MASK if i == 0 else cv2.GC_EVAL)
                if cancel:
                    cancel.raise_if_cancelled()
        except cv2.error as e:
            raise SelectionError("GrabCut נכשל על הקלט הזה") from e

        fg = np.where((mask == _FGD) | (mask == _PR_FGD), 255, 0).astype(np.uint8)
        fg = _keep_components(fg, pos_pts)
        if not fg.any():
            return None

        # --- חזרה לרזולוציה מלאה: חלון הניתוח (לפני הקטנת ה-ROI) → מלבן ברזולוציה מלאה, כמו בשרביט
        if (gh, gw) != (rh, rw):
            fg = cv2.resize(fg, (rw, rh), interpolation=cv2.INTER_LINEAR)
            fg = np.where(fg > 127, 255, 0).astype(np.uint8)
        x, y, bw, bh = cv2.boundingRect(fg)
        if bw == 0 or bh == 0:
            return None
        crop = fg[y:y + bh, x:x + bw]
        fx0, fy0 = min(round((rx0 + x) * sx), w - 1), min(round((ry0 + y) * sy), h - 1)
        fx1 = min(max(round((rx0 + x + bw) * sx), fx0 + 1), w)
        fy1 = min(max(round((ry0 + y + bh) * sy), fy0 + 1), h)
        if (fx1 - fx0, fy1 - fy0) != (bw, bh):
            crop = cv2.resize(crop, (fx1 - fx0, fy1 - fy0), interpolation=cv2.INTER_LINEAR)
        window = np.where(crop > 127, 255, 0).astype(np.uint8)
        if not window.any():
            return None
        return Proposal(window, (fx0, fy0, fx1, fy1))


def _keep_components(fg: np.ndarray, pos_pts: list[tuple[int, int]]) -> np.ndarray:
    """כל רכיב ששטחו לפחות `KEEP_RATIO` מהגדול ביותר, ובנוסף כל רכיב שמכיל נקודה חיובית. (חיבוריות 8.)"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return fg
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {1 + i for i in np.flatnonzero(areas >= KEEP_RATIO * areas.max()).tolist()}
    for qx, qy in pos_pts:
        lab = int(labels[qy, qx])
        if lab:
            keep.add(lab)
    return np.where(np.isin(labels, list(keep)), 255, 0).astype(np.uint8)


def _polygon_to_analysis(polygon, w: int, h: int, sx: float, sy: float) -> np.ndarray:
    """מצולע (קואורדינטות תמונה מלאה) → מערך int32 [N,2] בקואורדינטות הניתוח, חתוך לתמונה."""
    try:
        pts = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise SelectionError("מצולע לא תקין") from e
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise SelectionError("מצולע חייב לכלול לפחות 3 נקודות")
    if len(pts) > MAX_POLYGON_POINTS:
        raise SelectionError("יותר מדי נקודות במצולע")
    if not np.isfinite(pts).all():
        raise SelectionError("מצולע לא תקין")
    pts[:, 0] = np.clip(pts[:, 0], 0, w) / sx
    pts[:, 1] = np.clip(pts[:, 1], 0, h) / sy
    return np.round(pts).astype(np.int32)
