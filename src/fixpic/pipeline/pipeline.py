"""Pipeline: נרמול מסכה, קיבוץ, ROI או Tiling, Pad/Resize, הרצה, מיזוג (ARCHITECTURE §4, §8).
פיקסלים מחוץ למסכה האפקטיבית (אחרי הרחבה) זהים למקור לבייט, גם עם Feathering וגם עם Tiling."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..errors import EmptyMaskError, InvalidInputError
from ..models.base import CancelToken, InpaintModel
from .feather import feather_alpha
from .grouping import Group, group_mask
from .roi import RoiPlan, plan_roi_bbox, tile_cores, tile_plan


@dataclass(frozen=True)
class PipelineConfig:
    sizes: tuple[int, ...] = (256, 384, 512)   # buckets (ADR-011)
    default_size: int = 384
    mask_dilate_px: int = 5      # ADR-008; ערך זמני עד בדיקות איכות. 0 = כבוי
    margin_ratio: float = 0.25   # הקשר סביב המסכה; ערך זמני
    min_margin: int = 16
    feather_px: int = 3          # Feathering פנימי; ערך זמני. 0 = מיזוג בינארי
    # אזור שלא נכנס ב-bucket: "downscale" (הקטנה), "tile" (אריחים), "auto" (הקטנה עד min_scale, אחרת אריחים)
    large_strategy: str = "auto"
    min_scale: float = 0.5       # מתחתיו, ב-"auto", עוברים לאריחים; ערך זמני (איכות לא נמדדה)
    tile_overlap: int = 64       # הקשר סביב ליבת האריח
    max_tiles: int = 64          # מעל זה חוזרים להקטנה (תקרת זמן)


@dataclass(frozen=True)
class InpaintResult:
    image: np.ndarray            # תמונה חדשה; הקלט לא משתנה
    effective_mask: np.ndarray   # uint8 0/1 אחרי הרחבה; מחוץ לה הפיקסלים זהים למקור
    roi: RoiPlan                 # ה-ROI של הקבוצה הראשונה (תאימות לאחור)
    rois: list = field(default_factory=list)   # כל ה-ROI/האריחים שהורצו
    groups: int = 1
    model_calls: int = 0


class Pipeline:
    def __init__(self, model: InpaintModel, cfg: PipelineConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or PipelineConfig()
        if self.cfg.large_strategy not in ("auto", "tile", "downscale"):
            raise InvalidInputError(f"large_strategy לא תקף: {self.cfg.large_strategy}")
        for s in self.cfg.sizes:
            if s % model.info.multiple or not model.info.min_size <= s <= model.info.max_size:
                raise InvalidInputError(f"גודל bucket לא תקף למודל: {s}")

    def warmup(self, cancel: CancelToken | None = None) -> None:
        """compile מראש לכל ה-buckets (ברקע); הגודל הרגיל קודם. no-op ל-backend בלי warmup."""
        fn = getattr(self.model, "warmup", None)
        if fn:
            order = [self.cfg.default_size] + [s for s in self.cfg.sizes if s != self.cfg.default_size]
            fn(order, cancel)

    # ------------------------------------------------------------------ run
    def run(self, image: np.ndarray, mask: np.ndarray, size: int | None = None,
            cancel: CancelToken | None = None) -> InpaintResult:
        c = self.cfg
        size = size or c.default_size
        if size not in c.sizes:
            raise InvalidInputError(f"גודל קלט {size} אינו באחד ה-buckets {c.sizes}")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError("נדרשת תמונה uint8 בצורה [H,W,3] (RGB)")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise InvalidInputError("המסכה חייבת להיות [H,W] באותו גודל כמו התמונה")
        cancel = cancel or CancelToken()

        m = (mask > 0).astype(np.uint8)
        if not m.any():
            raise EmptyMaskError("לא סומן אזור")
        r = c.mask_dilate_px
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m, k)
        cancel.raise_if_cancelled()

        groups = group_mask(m, size, c.margin_ratio, c.min_margin)
        result = image.copy()                       # כשל באמצע אינו משנה את הקלט
        rois: list[RoiPlan] = []
        for g in groups:
            cancel.raise_if_cancelled()
            self._run_group(result, m, g, size, cancel, rois)
        return InpaintResult(result, m, rois[0], rois, len(groups), len(rois))

    # ---------------------------------------------------------------- group
    def _run_group(self, result, m, g: Group, size, cancel, rois) -> None:
        c = self.cfg
        hw = result.shape[:2]
        p = plan_roi_bbox(g.box, hw, size, c.margin_ratio, c.min_margin)
        use_tiles = False
        if p.downscaled and c.large_strategy != "downscale":
            cores = tile_cores(g.box, size, c.tile_overlap)
            if len(cores) <= c.max_tiles and (c.large_strategy == "tile" or p.scale < c.min_scale):
                use_tiles = True
        if not use_tiles:
            self._infer(result, m, g, p, (g.box), cancel, rois)
            return
        gm = Group(g.box, g.mask.copy())            # "נותר למילוי": ריק אחרי כל אריח
        for core in cores:
            cancel.raise_if_cancelled()
            x0, y0, x1, y1 = core
            bx, by = g.box[0], g.box[1]
            if not gm.mask[y0 - by:y1 - by, x0 - bx:x1 - bx].any():
                continue
            self._infer(result, m, gm, tile_plan(x0, y0, x1, y1, hw, size), core, cancel, rois)
            gm.mask[y0 - by:y1 - by, x0 - bx:x1 - bx] = 0

    # ---------------------------------------------------------------- infer
    def _infer(self, result, m, g: Group, p: RoiPlan, commit_box, cancel, rois) -> None:
        """מריץ חלון אחד. g.mask: מה שנותר למילוי (המודל רואה אותו); נכתבים רק פיקסלים של g.mask בתוך commit_box."""
        size = p.size
        win_mask = np.zeros((p.height, p.width), np.uint8)
        _paste(win_mask, g.mask, g.box, p)
        win_img = result[p.y0:p.y1, p.x0:p.x1]
        roi_img, roi_m = win_img, win_mask
        if p.downscaled:
            roi_img = cv2.resize(win_img, (p.in_w, p.in_h), interpolation=cv2.INTER_AREA)
            # max-pool דרך INTER_AREA וסף > 0, כדי לא לאבד פיקסלים דקים
            roi_m = (cv2.resize(win_mask, (p.in_w, p.in_h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        pad_b, pad_r = size - p.in_h, size - p.in_w
        if pad_b or pad_r:
            roi_img = cv2.copyMakeBorder(roi_img, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT)
            roi_m = cv2.copyMakeBorder(roi_m, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        cancel.raise_if_cancelled()

        out = self.model.run(np.ascontiguousarray(roi_img), np.ascontiguousarray(roi_m), cancel)
        if out.shape != (size, size, 3) or out.dtype != np.uint8:
            raise InvalidInputError(f"פלט המודל בצורה/טיפוס לא צפויים: {out.shape} {out.dtype}")
        out = out[:p.in_h, :p.in_w]
        if p.downscaled:
            out = cv2.resize(out, (p.width, p.height), interpolation=cv2.INTER_CUBIC)
        cancel.raise_if_cancelled()
        rois.append(p)

        # אזור הכתיבה: g.mask בתוך commit_box. alpha מחושב מהמסכה המלאה, כך שרק קצה המסכה האמיתי מעומעם
        # ולא גבולות האריחים. מחוץ למסכה alpha = 0.
        sel = np.zeros((p.height, p.width), bool)
        cx0, cy0, cx1, cy1 = (max(commit_box[0], p.x0) - p.x0, max(commit_box[1], p.y0) - p.y0,
                              min(commit_box[2], p.x1) - p.x0, min(commit_box[3], p.y1) - p.y0)
        sel[cy0:cy1, cx0:cx1] = win_mask[cy0:cy1, cx0:cx1] > 0
        alpha = self._alpha(m, p)
        a = np.where(sel, alpha, 0.0)[..., None]
        base = win_img.astype(np.float32)
        blended = np.rint(base + (out.astype(np.float32) - base) * a)
        win_img[sel] = blended.astype(np.uint8)[sel]   # win_img הוא view של result

    def _alpha(self, m, p: RoiPlan) -> np.ndarray:
        f = self.cfg.feather_px
        if f <= 0:
            return m[p.y0:p.y1, p.x0:p.x1].astype(np.float32)
        h, w = m.shape
        ex0, ey0, ex1, ey1 = max(0, p.x0 - f - 1), max(0, p.y0 - f - 1), min(w, p.x1 + f + 1), min(h, p.y1 + f + 1)
        al = feather_alpha(np.ascontiguousarray(m[ey0:ey1, ex0:ex1]), f)
        return al[p.y0 - ey0:p.y1 - ey0, p.x0 - ex0:p.x1 - ex0]


def _paste(dst: np.ndarray, src: np.ndarray, src_box, p: RoiPlan) -> None:
    """מדביק את src (ממוקם ב-src_box בקואורדינטות התמונה) לתוך dst (החלון p), עם חיתוך."""
    x0, y0 = max(src_box[0], p.x0), max(src_box[1], p.y0)
    x1, y1 = min(src_box[2], p.x1), min(src_box[3], p.y1)
    if x1 > x0 and y1 > y0:
        dst[y0 - p.y0:y1 - p.y0, x0 - p.x0:x1 - p.x0] = src[y0 - src_box[1]:y1 - src_box[1],
                                                           x0 - src_box[0]:x1 - src_box[0]]
