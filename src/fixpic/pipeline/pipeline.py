"""Pipeline: נרמול מסכה, ROI, Pad/Resize, הרצה, מיזוג (ARCHITECTURE §4).
פיקסלים מחוץ למסכה האפקטיבית (אחרי הרחבה) זהים למקור לבייט."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..errors import EmptyMaskError, InvalidInputError
from ..models.base import CancelToken, InpaintModel
from .roi import RoiPlan, plan_roi


@dataclass(frozen=True)
class PipelineConfig:
    sizes: tuple[int, ...] = (256, 384, 512)   # buckets (ADR-011)
    default_size: int = 384
    mask_dilate_px: int = 5      # ADR-008; ערך זמני עד בדיקות איכות. 0 = כבוי
    margin_ratio: float = 0.25   # הקשר סביב המסכה; ערך זמני
    min_margin: int = 16


@dataclass(frozen=True)
class InpaintResult:
    image: np.ndarray            # תמונה חדשה; הקלט לא משתנה
    effective_mask: np.ndarray   # uint8 0/1 אחרי הרחבה; מחוץ לה הפיקסלים זהים למקור
    roi: RoiPlan


class Pipeline:
    def __init__(self, model: InpaintModel, cfg: PipelineConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or PipelineConfig()
        for s in self.cfg.sizes:
            if s % model.info.multiple or not model.info.min_size <= s <= model.info.max_size:
                raise InvalidInputError(f"גודל bucket לא תקף למודל: {s}")

    def warmup(self, cancel: CancelToken | None = None) -> None:
        """compile מראש לכל ה-buckets (ברקע); הגודל הרגיל קודם. no-op ל-backend בלי warmup."""
        fn = getattr(self.model, "warmup", None)
        if fn:
            order = [self.cfg.default_size] + [s for s in self.cfg.sizes if s != self.cfg.default_size]
            fn(order, cancel)

    def run(self, image: np.ndarray, mask: np.ndarray, size: int | None = None,
            cancel: CancelToken | None = None) -> InpaintResult:
        size = size or self.cfg.default_size
        if size not in self.cfg.sizes:
            raise InvalidInputError(f"גודל קלט {size} אינו באחד ה-buckets {self.cfg.sizes}")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError("נדרשת תמונה uint8 בצורה [H,W,3] (RGB)")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise InvalidInputError("המסכה חייבת להיות [H,W] באותו גודל כמו התמונה")
        cancel = cancel or CancelToken()

        m = (mask > 0).astype(np.uint8)
        if not m.any():
            raise EmptyMaskError("לא סומן אזור")
        r = self.cfg.mask_dilate_px
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m, k)
        cancel.raise_if_cancelled()

        p = plan_roi(m, size, self.cfg.margin_ratio, self.cfg.min_margin)
        roi_img = image[p.y0:p.y1, p.x0:p.x1]
        roi_m = m[p.y0:p.y1, p.x0:p.x1]
        if p.downscaled:
            roi_img = cv2.resize(roi_img, (p.in_w, p.in_h), interpolation=cv2.INTER_AREA)
            # המסכה מוקטנת בלי לאבד פיקסלים דקים: max-pool דרך INTER_AREA ואז סף > 0
            roi_m = (cv2.resize(roi_m, (p.in_w, p.in_h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
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

        # מיזוג בינארי ברזולוציית המקור; הכתיבה לעותק, כך שכשל אינו משנה את הקלט
        result = image.copy()
        sel = m[p.y0:p.y1, p.x0:p.x1].astype(bool)
        view = result[p.y0:p.y1, p.x0:p.x1]
        view[sel] = out[sel]
        return InpaintResult(result, m, p)
