"""Pipeline: נרמול מסכה, קיבוץ, ROI או Tiling, Pad/Resize, הרצה, מיזוג (ARCHITECTURE §4, §8).
פיקסלים מחוץ למסכה האפקטיבית (אחרי הרחבה) זהים למקור לבייט, גם עם Feathering וגם עם Tiling."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from ..errors import EmptyMaskError, FixPicError, InferenceError, InvalidInputError, OutOfMemoryError, is_oom
from ..models.base import CancelToken, InpaintModel
from .dilate import dilate_radius, estimate_brush_px
from .feather import feather_alpha
from .grouping import Group, group_mask
from .roi import RoiPlan, plan_roi_bbox, tile_cores, tile_plan, tile_window

log = logging.getLogger("fixpic.pipeline")
STRIP = 128   # שורות בכל פעימת מיזוג: מונע מערכי float בגודל החלון כולו


@dataclass(frozen=True)
class PipelineConfig:
    sizes: tuple[int, ...] = (256, 384, 512)   # buckets (ADR-011)
    default_size: int = 384
    # ADR-008. None = אוטומטי: dilate_ratio × קוטר המברשת (brush_px, או מוערך מהמסכה), חסום בין dilate_min ל-dilate_max.
    # מספר = הרחבה קבועה בפיקסלים (0 = כבוי). כל הערכים זמניים עד בדיקת איכות על Big LaMa האמיתי.
    mask_dilate_px: int | None = None
    dilate_ratio: float = 0.125
    dilate_min: int = 3
    dilate_max: int = 20
    margin_ratio: float = 0.25   # הקשר סביב המסכה; ערך זמני
    min_margin: int = 16
    # §4ב (רזולוציה מקורית תחילה): לפני הקטנה, ההקשר מצטמצם עד יחס זה מהצלע הגדולה של המסכה (ולא פחות מ-min_margin)
    # כדי שהחלון יישאר בקנה מידה מקורי. שווה ל-margin_ratio = כבוי. ערך זמני, איכות לא נמדדה.
    native_margin_ratio: float = 0.10
    feather_px: int = 3          # Feathering פנימי; ערך זמני. 0 = מיזוג בינארי
    # אזור שלא נכנס ב-bucket: "downscale" (הקטנה), "tile" (אריחים), "auto" (הקטנה עד min_scale, אחרת אריחים)
    large_strategy: str = "auto"
    min_scale: float = 0.5       # מתחתיו, ב-"auto", עוברים לאריחים; ערך זמני (איכות לא נמדדה)
    tile_overlap: int = 64       # הקשר סביב ליבת האריח
    # M3.7.6: מיזוג הדרגתי בין אריחים. פס (בפיקסלי קלט למודל) לאורך הצדדים שבהם כבר מולא אריח שכן; הוא ממולא שוב
    # (המודל רואה אותו כמסכה) ומעורבב עם התוצאה הקודמת במשקל שעולה מ-0 ל-1. 0 = כבוי (מיזוג קשה בגבול הליבה). ערך זמני.
    tile_blend_px: int = 32
    # M3.7.7 (מגס לדק), רק לקבוצות שמעובדות באריחים: מעבר ראשון בחלון יחיד מוקטן (מבנה וצבע של כל האזור), ואז מעבר
    # עדין באריחים ברזולוציה מקורית (או הקטנה קלה) על חגורה לאורך גבול המסכה בלבד, עם התוצאה הגסה כהקשר. פנים האזור
    # נשאר כפי שהמעבר הגס מילא אותו. כבוי כברירת מחדל עד שיכויל על Big LaMa (M3.7.9). ערכים זמניים.
    coarse_fine: bool = False
    coarse_belt_px: int = 64     # רוחב החגורה, בפיקסלי קלט למודל (במקור: מוגדל פי 1/scale של האריחים)
    # bucket של המעבר הגס בלבד (חייב להיות ב-`sizes`); None = אותו bucket של האריחים. גדול יותר = פחות הקטנה ויותר הקשר.
    # בניסיון חוזר אחרי OOM חוזרים ל-bucket של האריחים.
    coarse_size: int | None = None
    max_tiles: int = 64          # מעל זה חוזרים להקטנה (תקרת זמן)
    memory_limit_mb: int | None = None   # תקרת זיכרון להערכה מראש (כולל המודל); None = ללא בדיקה
    model_overhead_mb: int = 1600        # שיא תהליך של OpenVINO נמדד ב-M1.5: 1.51GB; ערך מעוגל כלפי מעלה
    oom_retry: bool = True               # בכשל זיכרון: ניסיון חוזר ב-bucket קטן יותר


@dataclass(frozen=True)
class Progress:
    done: int       # חלונות (הרצות מודל) שהושלמו
    total: int      # סך החלונות המתוכננים; יכול לגדול אחרי ניסיון חוזר ב-OOM
    stage: str      # "inpaint" | "done"


ProgressFn = Callable[[Progress], None]


def estimate_memory_mb(h: int, w: int, model_overhead_mb: int = 0) -> int:
    """הערכה מראש של שיא הזיכרון של עיבוד תמונה h×w (מעל התמונה שהמתקשר מחזיק), כולל המודל.
    מקדם הבתים לפיקסל מכויל בבדיקה (tests/test_m25.py) מול tracemalloc, עם מרווח ביטחון."""
    return int(model_overhead_mb + BYTES_PER_PIXEL * h * w / 2**20) + 1


BYTES_PER_PIXEL = 32      # נמדד: עד 28.4 בייט/פיקסל בעיבוד מסכה ענקית בהקטנה (tracemalloc, 12MP ו-24MP)


@dataclass(frozen=True)
class InpaintResult:
    image: np.ndarray            # תמונה חדשה; הקלט לא משתנה
    effective_mask: np.ndarray   # uint8 0/1 אחרי הרחבה; מחוץ לה הפיקסלים זהים למקור
    roi: RoiPlan                 # ה-ROI של הקבוצה הראשונה (תאימות לאחור)
    rois: list = field(default_factory=list)   # כל ה-ROI/האריחים שהורצו
    groups: int = 1
    model_calls: int = 0
    size_used: int = 0           # ה-bucket שבו הושלם הקבוצה האחרונה (קטן מהמבוקש אחרי OOM)
    oom_retries: int = 0
    dilate_px: int = 0           # רדיוס ההרחבה שבוצעה בפועל
    brush_px: float = 0.0        # קוטר המברשת ששימש לחישובה (0 = הרחבה קבועה, לא נדרש)
    feather_px: int = 0          # רוחב ה-Feathering שבו נעשה שימוש (כולל עקיפה לפי עבודה)


@dataclass(frozen=True)
class GroupPlan:
    box: tuple[int, int, int, int]
    mask_px: int
    kind: str                   # "native" | "downscale" | "tile"
    scale: float                # קנה מידה בפועל של הפיקסלים שנשלחים למודל (1.0 = מקורי)
    window: tuple[int, int]     # רוחב, גובה החלון בפיקסלי המקור (ב-"tile": חלון הקבוצה כולה, לא האריח)
    windows: int                # הרצות מודל


@dataclass(frozen=True)
class PlanReport:
    size: int
    dilate_px: int
    brush_px: float
    mask_px: int                # אחרי הרחבה
    groups: list[GroupPlan]

    @property
    def runs(self) -> int:
        return sum(g.windows for g in self.groups)

    @property
    def min_scale(self) -> float:
        return min((g.scale for g in self.groups), default=1.0)


class Pipeline:
    def __init__(self, model: InpaintModel, cfg: PipelineConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or PipelineConfig()
        self.coarse_hook: Callable[[tuple[int, int, int, int], np.ndarray], None] | None = None   # אבחון M3.7.7
        if self.cfg.large_strategy not in ("auto", "tile", "downscale"):
            raise InvalidInputError(f"large_strategy לא תקף: {self.cfg.large_strategy}")
        c = self.cfg
        if (c.dilate_ratio < 0 or c.dilate_min < 0 or c.dilate_max < c.dilate_min
                or (c.mask_dilate_px is not None and c.mask_dilate_px < 0)
                or not 0 <= c.native_margin_ratio <= c.margin_ratio):
            raise InvalidInputError("פרמטרי הרחבה/הקשר לא תקינים")
        for s in self.cfg.sizes:
            if s % model.info.multiple or not model.info.min_size <= s <= model.info.max_size:
                raise InvalidInputError(f"גודל bucket לא תקף למודל: {s}")

    def warmup_order(self) -> list[int]:
        """סדר ה-buckets ל-Warm-up: הגודל הרגיל קודם."""
        return [self.cfg.default_size] + [s for s in self.cfg.sizes if s != self.cfg.default_size]

    def warmup(self, cancel: CancelToken | None = None, sizes: list[int] | None = None) -> None:
        """compile מראש ל-buckets (ברקע); ברירת מחדל: כולם, לפי warmup_order. no-op ל-backend בלי warmup."""
        fn = getattr(self.model, "warmup", None)
        if fn:
            fn(sizes if sizes is not None else self.warmup_order(), cancel)

    # ------------------------------------------------------------------ run
    def run(self, image: np.ndarray, mask: np.ndarray, size: int | None = None,
            cancel: CancelToken | None = None, progress: ProgressFn | None = None,
            brush_px: float | None = None, feather_px: int | None = None) -> InpaintResult:
        """brush_px: קוטר המברשת בפיקסלי התמונה (מה-UI), לחישוב ההרחבה האוטומטית; None = הערכה מהמסכה.
        feather_px: עקיפה של `cfg.feather_px` לריצה זו (0 = מיזוג בינארי, עד `MAX_FEATHER`); None = לפי ה-cfg.

        גבול שכבת ה-Engine (ARCHITECTURE §10): רק `FixPicError` יוצא החוצה. חריגה לא צפויה מומרת
        ל-`InferenceError` (או ל-`OutOfMemoryError` אם זו חריגת זיכרון), והקלט לעולם אינו משתנה."""
        try:
            return self._run(image, mask, size, cancel, progress, brush_px, feather_px)
        except FixPicError:
            raise
        except Exception as e:  # noqa: BLE001 - ממיר חריגות לא צפויות לשגיאה מוגדרת
            if is_oom(e):
                raise OutOfMemoryError(f"נגמר הזיכרון: {e}") from e
            log.exception("חריגה לא צפויה ב-Pipeline")
            raise InferenceError(f"שגיאה לא צפויה: {e}") from e

    def _run(self, image, mask, size, cancel, progress, brush_px, feather_px=None) -> InpaintResult:
        c = self.cfg
        feather = check_feather(c.feather_px if feather_px is None else feather_px)
        size = size or c.default_size
        if size not in c.sizes:
            raise InvalidInputError(f"גודל קלט {size} אינו באחד ה-buckets {c.sizes}")
        if c.coarse_fine and c.coarse_size is not None and c.coarse_size not in c.sizes:
            raise InvalidInputError(f"coarse_size {c.coarse_size} אינו באחד ה-buckets {c.sizes}")
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError("נדרשת תמונה uint8 בצורה [H,W,3] (RGB)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise InvalidInputError("התמונה ריקה (גובה או רוחב 0)")
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise InvalidInputError("המסכה חייבת להיות [H,W] באותו גודל כמו התמונה")
        cancel = cancel or CancelToken()
        if c.memory_limit_mb is not None:
            need = estimate_memory_mb(*image.shape[:2], c.model_overhead_mb)
            if need > c.memory_limit_mb:
                raise OutOfMemoryError(f"הערכת זיכרון {need}MB חורגת מהתקרה {c.memory_limit_mb}MB; התמונה לא שונתה")

        m, r, brush_px = self._prepare_mask(mask, brush_px)
        cancel.raise_if_cancelled()

        hw = image.shape[:2]
        try:
            groups = group_mask(m, size, c.margin_ratio, c.min_margin)
        except MemoryError as e:
            raise OutOfMemoryError(f"נגמר הזיכרון בניתוח המסכה: {e}") from e
        result = image.copy()                       # כשל באמצע אינו משנה את הקלט
        rois: list[RoiPlan] = []
        st = _State(progress, sum(self._count(g, size, hw) for g in groups))
        used = size
        for g in groups:
            cancel.raise_if_cancelled()
            used = self._run_group_with_retry(image, result, m, g, size, cancel, rois, st, hw, feather)
        st.emit("done")
        return InpaintResult(result, m, rois[0], rois, len(groups), len(rois), used, st.retries, r,
                             float(brush_px or 0.0), feather)

    def _prepare_mask(self, mask: np.ndarray, brush_px: float | None) -> tuple[np.ndarray, int, float | None]:
        """מסכה בינארית אחרי הרחבה: (m, רדיוס ההרחבה, brush_px בפועל)."""
        c = self.cfg
        m = (mask > 0).astype(np.uint8)
        if not m.any():
            raise EmptyMaskError("לא סומן אזור")
        if c.mask_dilate_px is None and brush_px is None:
            brush_px = estimate_brush_px(m)
        r = dilate_radius(c.mask_dilate_px, brush_px or 0.0, c.dilate_ratio, c.dilate_min, c.dilate_max)
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            m = cv2.dilate(m, k)
        return m, r, brush_px

    def plan(self, mask: np.ndarray, size: int | None = None, brush_px: float | None = None) -> "PlanReport":
        """מה ש-`run` יתכנן למסכה זו, בלי להריץ מודל ובלי תמונה (אבחון, M3.7.2): הרחבה, קבוצות, וכל קבוצה: חלון,
        קנה מידה, אסטרטגיה (native/downscale/tile) ומספר הרצות מודל. משתמש באותו קוד תכנון כמו `run`."""
        c = self.cfg
        size = size or c.default_size
        if size not in c.sizes:
            raise InvalidInputError(f"גודל קלט {size} אינו באחד ה-buckets {c.sizes}")
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise InvalidInputError("המסכה חייבת להיות [H,W]")
        m, r, brush_px = self._prepare_mask(mask, brush_px)
        hw = m.shape
        out = []
        for g in group_mask(m, size, c.margin_ratio, c.min_margin):
            p, cores, ts = self._strategy(g, size, hw)
            if cores is not None:
                runs = self._count(g, size, hw)
                kind, eff, windows = "tile", ts, runs
            elif p.downscaled:
                kind, eff, windows = "downscale", p.scale, 1
            else:
                kind, eff, windows = "native", 1.0, 1
            out.append(GroupPlan(g.box, int((g.mask > 0).sum()), kind, eff, (p.width, p.height), windows))
        return PlanReport(size, r, float(brush_px or 0.0), int((m > 0).sum()), out)

    def tile_seams(self, mask: np.ndarray, size: int | None = None, brush_px: float | None = None,
                   min_run: int = 16) -> list[tuple[str, int, int, int]]:
        """קווי התפר בין אריחים (M3.7.4), בפורמט של `quality.seam_ratio`: `("v", x, y0, y1)` ו-`("h", y, x0, x1)`.
        רק קטעים שבהם המסכה האפקטיבית מכסה את שני צדי הקו (שם באמת נוצר תפר) ואורכם לפחות `min_run`. בלי הרצת מודל;
        ריק כשאין אריחים. משתמש באותו תכנון כמו `run`/`plan`."""
        size = size or self.cfg.default_size
        m, _r, _b = self._prepare_mask(mask, brush_px)
        hw = m.shape
        lines: list[tuple[str, int, int, int]] = []

        def runs(ok: np.ndarray, off: int) -> list[tuple[int, int]]:
            d = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
            return [(off + int(a), off + int(b)) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])
                    if b - a >= min_run]

        for g in group_mask(m, size, self.cfg.margin_ratio, self.cfg.min_margin):
            _p, cores, _ts = self._strategy(g, size, hw)
            if cores is None:
                continue
            gx0, gy0, gx1, gy1 = g.box
            cov = m[gy0:gy1, gx0:gx1]
            if self.cfg.coarse_fine:        # התפרים הם רק של המעבר העדין, כלומר בתוך החגורה
                cov = self._belt(g, _ts, hw)[0]
            for x in sorted({c[0] for c in cores} | {c[2] for c in cores}):
                if gx0 < x < gx1:
                    both = (cov[:, x - 1 - gx0] > 0) & (cov[:, x - gx0] > 0)
                    lines += [("v", x, a, b) for a, b in runs(both, gy0)]
            for y in sorted({c[1] for c in cores} | {c[3] for c in cores}):
                if gy0 < y < gy1:
                    both = (cov[y - 1 - gy0, :] > 0) & (cov[y - gy0, :] > 0)
                    lines += [("h", y, a, b) for a, b in runs(both, gx0)]
        return lines

    def _count(self, g: Group, size: int, hw) -> int:
        """מספר החלונות שהקבוצה תדרוש, לצורך total בדיווח התקדמות."""
        _, cores, ts = self._strategy(g, size, hw)
        if cores is None:
            return 1
        if self.cfg.coarse_fine:
            return 1 + self._count_tiles(self._belt(g, ts, hw)[0], g.box, cores)
        return self._count_tiles(g.mask, g.box, cores)

    @staticmethod
    def _count_tiles(mask, box, cores) -> int:
        bx, by = box[0], box[1]
        return sum(1 for x0, y0, x1, y1 in cores if mask[y0 - by:y1 - by, x0 - bx:x1 - bx].any())

    def _belt(self, g: Group, ts: float, hw) -> tuple[np.ndarray, np.ndarray]:
        """M3.7.7: (חגורה, gain) בגודל ה-box. חגורה: פיקסלי המסכה במרחק עד `coarse_belt_px` (במקור) מהגבול האמיתי של
        הקבוצה. ה-box נחשב גבול (מחוץ לו אין מסכה), חוץ מצדדים שנוגעים בקצה התמונה (שם אין גבול). gain: 1 בעומק החגורה
        ויורד ל-0 בקצה הפנימי שלה (חצי מהרוחב), כדי שהמעבר העדין יתמזג בהדרגה בתוצאה הגסה."""
        k = 1.0 / ts if ts < 1.0 else 1.0
        belt = max(2, math.ceil(self.cfg.coarse_belt_px * k))
        x0, y0, x1, y1 = g.box
        h, w = hw
        pad = cv2.copyMakeBorder(g.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        if y0 == 0:
            pad[0, :] = 1
        if y1 == h:
            pad[-1, :] = 1
        if x0 == 0:
            pad[:, 0] = 1
        if x1 == w:
            pad[:, -1] = 1
        dist = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
        ramp = max(1.0, belt / 2.0)
        gain = np.clip((belt - dist) / ramp, 0.0, 1.0).astype(np.float32)
        return ((dist <= belt) & (g.mask > 0)).astype(np.uint8), gain

    def _run_group_with_retry(self, image, result, m, g: Group, size, cancel, rois, st, hw, feather) -> int:
        sizes = sorted((s for s in self.cfg.sizes if s <= size), reverse=True)
        n_before = len(rois)
        for i, s in enumerate(sizes):
            try:
                self._run_group(result, m, g, s, cancel, rois, st, feather, self.cfg.coarse_size if i == 0 else None)
                return s
            except (OutOfMemoryError, MemoryError) as e:
                if not is_oom(e):
                    raise
                # מחזירים את פיקסלי הקבוצה למקור (רק הם שונו על ידה) ומנסים bucket קטן יותר
                sel = g.mask > 0
                x0, y0, x1, y1 = g.box
                result[y0:y1, x0:x1][sel] = image[y0:y1, x0:x1][sel]
                del rois[n_before:]
                if not self.cfg.oom_retry or i + 1 >= len(sizes):
                    raise e if isinstance(e, OutOfMemoryError) else OutOfMemoryError(f"נגמר הזיכרון: {e}") from e
                log.warning("OOM ב-bucket %d, ניסיון חוזר ב-%d", s, sizes[i + 1])
                st.retries += 1
                st.total += self._count(g, sizes[i + 1], hw)
        raise AssertionError("unreachable")

    # ---------------------------------------------------------------- group
    def _tile_scale(self, box, size: int, floor: float) -> float | None:
        """קנה המידה הגדול ביותר (עד 1.0 = מקורי) שבו אריחים מכסים את `box` בלא יותר מ-`max_tiles` הרצות, או None אם
        גם ב-`floor` (קנה המידה של חלון יחיד) זה לא מתקיים. מספר האריחים אינו עולה כשהקנה קטן, ולכן חיפוש בינארי.
        כך שאזור גדול מאוד מוקטן בהדרגה, ולא ב"מצוק" מקנה מידה מקורי להקטנה פי 16 בחלון אחד (M3.7.6)."""
        c = self.cfg
        n = lambda s: len(tile_cores(box, size, c.tile_overlap, s))      # noqa: E731
        if n(1.0) <= c.max_tiles:
            return 1.0
        if n(floor) > c.max_tiles:
            return None
        lo, hi = floor, 1.0                                            # n(lo) <= max < n(hi)
        for _ in range(24):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if n(mid) <= c.max_tiles else (lo, mid)
        return lo

    def _strategy(self, g: Group, size: int, hw):
        """(תוכנית ה-ROI, ליבות אריחים או None, קנה מידה של האריחים).
        `downscale`: חלון יחיד מוקטן. `tile`: אריחים (מקוריים אם נכנסים ב-`max_tiles`, אחרת מוקטנים כמה שצריך).
        `auto`: חלון יחיד כל עוד קנה המידה שלו לפחות `min_scale`; אחרת אריחים, ובלבד שקנה המידה שלהם טוב מזה של חלון יחיד."""
        c = self.cfg
        p = plan_roi_bbox(g.box, hw, size, c.margin_ratio, c.min_margin, c.native_margin_ratio)
        if p.downscaled and c.large_strategy != "downscale" and (c.large_strategy == "tile" or p.scale < c.min_scale):
            ts = self._tile_scale(g.box, size, p.scale)
            if ts is not None and (ts > p.scale or c.large_strategy == "tile"):
                return p, tile_cores(g.box, size, c.tile_overlap, ts), ts
        return p, None, p.scale

    def _run_group(self, result, m, g: Group, size, cancel, rois, st, feather, coarse_size=None) -> None:
        hw = result.shape[:2]
        p, cores, ts = self._strategy(g, size, hw)
        if cores is None:
            self._infer(result, m, g, p, g.box, cancel, rois, feather)
            st.step()
            return
        if not self.cfg.coarse_fine:
            self._run_tiles(result, m, g, size, cores, ts, cancel, rois, st, feather)
            return
        # מגס לדק: (1) חלון יחיד מוקטן ממלא את כל האזור (ב-`coarse_size` אם הוגדר); (2) אריחים רק על החגורה, והמודל
        # רואה את המילוי הגס כהקשר
        pc = p
        if coarse_size and coarse_size != size:
            c = self.cfg
            pc = plan_roi_bbox(g.box, hw, coarse_size, c.margin_ratio, c.min_margin, c.native_margin_ratio)
        self._infer(result, m, g, pc, g.box, cancel, rois, feather)
        st.step()
        if self.coarse_hook is not None:    # אבחון: העתק של תיבת הקבוצה אחרי המעבר הגס בלבד
            x0, y0, x1, y1 = g.box
            self.coarse_hook(g.box, result[y0:y1, x0:x1].copy())
        belt, gain = self._belt(g, ts, hw)
        self._run_tiles(result, m, Group(g.box, belt), size, cores, ts, cancel, rois, st, feather, m_alpha=m, gain=gain)

    def _run_tiles(self, result, m, g: Group, size, cores, ts, cancel, rois, st, feather, m_alpha=None, gain=None) -> None:
        """אריחים על g.mask. `m_alpha`: המסכה שלפיה מחושב ה-Feathering של הקצה (ברירת מחדל: m); `gain`: משקל נוסף בגודל
        ה-box של הקבוצה (המעבר העדין של מגס-לדק)."""
        hw = result.shape[:2]
        gm = Group(g.box, g.mask.copy())            # "נותר למילוי": ריק אחרי כל אריח
        bx, by = g.box[0], g.box[1]
        c = self.cfg
        k = 1.0 / ts if ts < 1.0 else 1.0
        blend = min(math.ceil(c.tile_blend_px * k), math.ceil(c.tile_overlap * k)) if c.tile_blend_px > 0 else 0
        for core in cores:
            cancel.raise_if_cancelled()
            x0, y0, x1, y1 = core
            if not gm.mask[y0 - by:y1 - by, x0 - bx:x1 - bx].any():
                continue
            # פס מיזוג: רק לכיוונים שכבר מולאו (שמאל ולמעלה, סדר שורות), ולא מעבר לתיבת הקבוצה
            ex0 = max(bx, x0 - blend) if x0 > bx else x0
            ey0 = max(by, y0 - blend) if y0 > by else y0
            ebox = (ex0, ey0, x1, y1)
            weight = None
            if ex0 < x0 or ey0 < y0:
                wx = np.ones(x1 - ex0, np.float32)
                wx[:x0 - ex0] = (np.arange(x0 - ex0, dtype=np.float32) + 1) / (x0 - ex0 + 1)
                wy = np.ones(y1 - ey0, np.float32)
                wy[:y0 - ey0] = (np.arange(y0 - ey0, dtype=np.float32) + 1) / (y0 - ey0 + 1)
                weight = np.minimum(wy[:, None], wx[None, :])
                # בפס, המודל רואה את המסכה המקורית (ולא את "נותר למילוי") כדי למלא אותו מחדש
                gm.mask[ey0 - by:y1 - by, ex0 - bx:x1 - bx] = g.mask[ey0 - by:y1 - by, ex0 - bx:x1 - bx]
            self._infer(result, m if m_alpha is None else m_alpha, gm, tile_plan(ex0, ey0, x1, y1, hw, size, ts), ebox,
                        cancel, rois, feather, weight, None if gain is None else (gain, (bx, by)))
            gm.mask[ey0 - by:y1 - by, ex0 - bx:x1 - bx] = 0
            st.step()

    # ---------------------------------------------------------------- infer
    def _infer(self, result, m, g: Group, p: RoiPlan, commit_box, cancel, rois, feather, weight=None, gain=None) -> None:
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

        try:
            out = self.model.run(np.ascontiguousarray(roi_img), np.ascontiguousarray(roi_m), cancel)
        except MemoryError as e:
            raise OutOfMemoryError(f"נגמר הזיכרון בהרצת המודל: {e}") from e
        if out.shape != (size, size, 3) or out.dtype != np.uint8:
            raise InvalidInputError(f"פלט המודל בצורה/טיפוס לא צפויים: {out.shape} {out.dtype}")
        out = out[:p.in_h, :p.in_w]
        if p.downscaled:
            out = cv2.resize(out, (p.width, p.height), interpolation=cv2.INTER_CUBIC)
        cancel.raise_if_cancelled()
        rois.append(p)

        # אזור הכתיבה: g.mask בתוך commit_box. alpha מחושב מהמסכה המלאה, כך שרק קצה המסכה האמיתי מעומעם
        # ולא גבולות האריחים. מחוץ למסכה alpha = 0. המיזוג בפסים כדי לא להחזיק float32 בגודל החלון.
        cx0, cy0, cx1, cy1 = (max(commit_box[0], p.x0) - p.x0, max(commit_box[1], p.y0) - p.y0,
                              min(commit_box[2], p.x1) - p.x0, min(commit_box[3], p.y1) - p.y0)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        region = (p.x0 + cx0, p.y0 + cy0, p.x0 + cx1, p.y0 + cy1)
        alpha = self._alpha(m, region, feather)
        if weight is not None:      # משקל מיזוג בין אריחים (ממוקם ב-commit_box; region יכול להיות חיתוך שלו)
            wy0, wx0 = region[1] - commit_box[1], region[0] - commit_box[0]
            alpha = alpha * weight[wy0:wy0 + alpha.shape[0], wx0:wx0 + alpha.shape[1]]
        if gain is not None:        # משקל נוסף בגודל ה-box של הקבוצה, במקום (bx, by)
            ga, (gbx, gby) = gain
            alpha = alpha * ga[region[1] - gby:region[3] - gby, region[0] - gbx:region[2] - gbx]
        for r0 in range(cy0, cy1, STRIP):
            r1 = min(cy1, r0 + STRIP)
            sel = win_mask[r0:r1, cx0:cx1] > 0
            if not sel.any():
                continue
            dst = win_img[r0:r1, cx0:cx1]                      # view של result
            a = np.where(sel, alpha[r0 - cy0:r1 - cy0], 0.0)[..., None]
            base = dst.astype(np.float32)
            blended = np.rint(base + (out[r0:r1, cx0:cx1].astype(np.float32) - base) * a).astype(np.uint8)
            dst[sel] = blended[sel]

    def _alpha(self, m, region, f: int) -> np.ndarray:
        x0, y0, x1, y1 = region
        if f <= 0:
            return m[y0:y1, x0:x1].astype(np.float32)
        h, w = m.shape
        ex0, ey0, ex1, ey1 = max(0, x0 - f - 1), max(0, y0 - f - 1), min(w, x1 + f + 1), min(h, y1 + f + 1)
        al = feather_alpha(np.ascontiguousarray(m[ey0:ey1, ex0:ex1]), f)
        return al[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]


class _State:
    """מונה התקדמות. חריגה ב-callback אינה מפילה את העיבוד (נרשמת ביומן)."""

    def __init__(self, fn: ProgressFn | None, total: int) -> None:
        self.fn, self.total, self.done, self.retries = fn, total, 0, 0

    def step(self) -> None:
        self.done += 1
        self.emit("inpaint")

    def emit(self, stage: str) -> None:
        if self.fn:
            try:
                self.fn(Progress(self.done, max(self.total, self.done), stage))
            except Exception:  # noqa: BLE001
                log.exception("callback של התקדמות נכשל")


MAX_FEATHER = 64   # תקרת הגיוניות לרוחב Feathering (פיקסלי תמונה); מעבר לכך אין תועלת והחישוב מיותר


def check_feather(f) -> int:
    if isinstance(f, bool) or not isinstance(f, (int, np.integer)) or not 0 <= f <= MAX_FEATHER:
        raise InvalidInputError(f"feather_px חייב להיות מספר שלם בין 0 ל-{MAX_FEATHER}")
    return int(f)


def _paste(dst: np.ndarray, src: np.ndarray, src_box, p: RoiPlan) -> None:
    """מדביק את src (ממוקם ב-src_box בקואורדינטות התמונה) לתוך dst (החלון p), עם חיתוך."""
    x0, y0 = max(src_box[0], p.x0), max(src_box[1], p.y0)
    x1, y1 = min(src_box[2], p.x1), min(src_box[3], p.y1)
    if x1 > x0 and y1 > y0:
        dst[y0 - p.y0:y1 - p.y0, x0 - p.x0:x1 - p.x0] = src[y0 - src_box[1]:y1 - src_box[1],
                                                           x0 - src_box[0]:x1 - src_box[0]]
