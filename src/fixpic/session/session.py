"""Session (ARCHITECTURE §4א, §9, §10): התמונה העובדת, שכבת המסכה וההיסטוריה של עריכה אחת. ללא UI.

חוזים:
- **Thread אחד.** כל הקריאות ל-Session (כולל `finish`) מה-thread של ה-UI. ה-callbacks של ה-Scheduler רצים ב-thread
  של ה-Worker ויכולים רק להעביר הודעה ל-UI (ב-Qt: signal), וה-UI קורא ל-`finish(job)`.
- **Commit אטומי ו-Copy-on-Write.** התמונה העובדת אינה משתנה במקום: כל שינוי יוצר מערכים חדשים (קריאה בלבד)
  ומחליף הפניה, כי Jobs ממתינים ותצוגה מחזיקים את המערך הקודם. עלות (נמדד בסביבת הפיתוח, חציון, 24MP): Commit
  כ-50ms למשיכה קטנה (כ-0.2 שנייה עם Alpha) ועד כ-0.5 שנייה למסכה של 4MP; Undo/Redo כ-40-180ms.
  כל מה שעלול להיכשל (זיכרון, בניית Patch) קורה לפני ההחלפה, ולכן כשל משאיר את התמונה וההיסטוריה כפי שהיו.
- **Commit לפי מסכה, לא החלפת תמונה.** Job שהוגש לפני Commit של Job קודם רץ על תמונה ישנה; הדבקת כל תוצאתו
  הייתה מוחקת את השינוי הקודם. לכן מודבקים רק פיקסלי המסכה האפקטיבית, ו-Patch נבנה מהמצב הנוכחי (ולא ממצב
  הבסיס של ה-Job), כך שרצף Patches תמיד עקבי.
- **Undo בזמן עיבוד = ביטול** (PARITY-SPEC §3.4): מבטל את הרץ ומרוקן את התור, בלי לגעת בהיסטוריה. Job שכבר
  הסתיים אך טרם נקלט ב-`finish` מזוהה כמיושן (epoch) ואינו נכתב.
- **Alpha:** ראו `alpha.py`. ב-Undo/Redo ה-Alpha חוזר יחד עם ה-RGB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..engine import Job, JobScheduler, JobState
from ..errors import (EmptyMaskError, FixPicError, InvalidInputError, OutOfMemoryError)
from ..history import DEFAULT_MAX_BYTES, DEFAULT_MAX_STEPS, History, Patch
from ..imgio import LoadedImage, SaveResult, save_image
from ..pipeline import InpaintResult, Progress
from .alpha import fill_alpha
from .mask_layer import MaskLayer

log = logging.getLogger("fixpic.session")


class OutcomeKind(str, Enum):
    COMMITTED = "committed"   # התמונה עודכנה (Patch חדש בהיסטוריה), או שלא היה שינוי בפועל (`patch` = None)
    FAILED = "failed"         # העבודה נכשלה או ה-Commit נכשל; התמונה לא השתנתה והמסכה הוחזרה לשכבה
    CANCELLED = "cancelled"   # בוטלה ישירות (`Job.cancel`); המסכה הוחזרה לשכבה
    STALE = "stale"           # בוטלה ע"י Undo (או אינה של ה-Session); התמונה והמסכה לא נגעו


@dataclass(frozen=True)
class JobOutcome:
    kind: OutcomeKind
    error: BaseException | None = None
    patch: Patch | None = None


class HistoryAction(str, Enum):
    UNDONE = "undone"
    REDONE = "redone"
    CANCELLED_JOBS = "cancelled_jobs"   # Undo בזמן עיבוד: בוטלו Jobs; ההיסטוריה לא זזה
    BUSY = "busy"                       # Redo בזמן עיבוד: נדחה (משיכה חדשה ממילא תאפס את Redo)
    NOTHING = "nothing"                 # אין מה לבטל/לשחזר


@dataclass(frozen=True)
class _Pending:
    mask: np.ndarray
    epoch: int


def _frozen(a: np.ndarray | None) -> np.ndarray | None:
    if a is not None:
        a.setflags(write=False)
    return a


class Session:
    def __init__(self, image: LoadedImage, scheduler: JobScheduler | None = None, *,
                 max_steps: int = DEFAULT_MAX_STEPS, max_history_bytes: int = DEFAULT_MAX_BYTES) -> None:
        """לוקח בעלות על מערכי `image` (הם מסומנים כקריאה בלבד ואין לשנותם). `scheduler` אופציונלי: בלעדיו אפשר
        להשתמש ב-`commit` ישירות (CLI, בדיקות)."""
        rgb = image.rgb
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.size == 0):
            raise InvalidInputError("נדרשת תמונה uint8 בצורה [H,W,3]")
        if image.alpha is not None and (image.alpha.dtype != np.uint8 or image.alpha.shape != rgb.shape[:2]):
            raise InvalidInputError("Alpha חייב להיות uint8 בגודל התמונה")
        self._rgb = _frozen(np.ascontiguousarray(rgb))
        self._alpha = _frozen(None if image.alpha is None else np.ascontiguousarray(image.alpha))
        self._orig_rgb, self._orig_alpha = self._rgb, self._alpha      # אותם מערכים: אין עותק נוסף עד ה-Commit הראשון
        self.icc_profile, self.exif = image.icc_profile, image.exif
        self.source_path, self.source_format = image.source_path, image.source_format
        self.scheduler = scheduler
        self.mask = MaskLayer(*rgb.shape[:2])
        self.history = History(max_steps, max_history_bytes)
        self.version = 0                    # עולה בכל שינוי בתמונה (Commit/Undo/Redo)
        self._epoch = 0                     # עולה ב-Undo/Redo: Jobs מ-epoch ישן אינם נכתבים
        self._pending: dict[Job, _Pending] = {}
        self._saved_id = 0

    # ----------------------------------------------------------------- מצב
    @property
    def rgb(self) -> np.ndarray:
        return self._rgb

    @property
    def alpha(self) -> np.ndarray | None:
        return self._alpha

    @property
    def original_rgb(self) -> np.ndarray:
        """התמונה כפי שנטענה (להשוואה לפני/אחרי)."""
        return self._orig_rgb

    @property
    def original_alpha(self) -> np.ndarray | None:
        return self._orig_alpha

    @property
    def size(self) -> tuple[int, int]:
        """(רוחב, גובה)."""
        return self._rgb.shape[1], self._rgb.shape[0]

    @property
    def busy(self) -> bool:
        """יש משיכות שהוגשו וטרם נקלטו ב-`finish`."""
        return bool(self._pending)

    @property
    def pending_masks(self) -> list[np.ndarray]:
        """מסכות של משיכות בעיבוד (להצגה מעל התמונה בזמן Processing), בסדר ההגשה."""
        return [p.mask for p in self._pending.values()]

    @property
    def modified(self) -> bool:
        """התמונה שונה מאז הטעינה/השמירה האחרונה (Undo חזרה למצב השמור = False)."""
        return self.history.current_id != self._saved_id

    def mark_saved(self) -> None:
        self._saved_id = self.history.current_id

    def save(self, path: str | Path, **kw) -> SaveResult:
        """שמירה אטומית ברזולוציה מלאה עם ה-ICC/EXIF שנטענו (kw: fmt, quality, lossless). מסמנת "נשמר" בהצלחה."""
        res = save_image(path, self._rgb, self._alpha, icc_profile=self.icc_profile, exif=self.exif, **kw)
        self.mark_saved()
        return res

    # ------------------------------------------------------- משיכה -> Job -> Commit
    def submit_stroke(self, *, size: int | None = None, brush_px: float | None = None, feather_px: int | None = None,
                      on_progress: Callable[[Job, Progress], None] | None = None,
                      on_done: Callable[[Job], None] | None = None) -> Job:
        """שחרור מברשת: מגיש את מסכת השכבה כ-Job על התמונה העובדת ומרוקן את השכבה (המשיכה הבאה מתחילה נקייה).
        EmptyMaskError אם השכבה ריקה; QueueFullError/SchedulerClosedError מה-Scheduler, ובשניהם המסכה נשארת
        בשכבה. `on_done` רץ ב-thread של ה-Worker (ראו חוזים); בסיומו ה-UI קורא ל-`finish`."""
        if self.scheduler is None:
            raise InvalidInputError("ל-Session אין Scheduler")
        if self.mask.empty:
            raise EmptyMaskError("לא סומן אזור")
        mask = self.mask.snapshot()
        brush = brush_px if brush_px is not None else self.mask.brush_px
        job = self.scheduler.submit(self._rgb, mask, size, on_progress, on_done, brush, feather_px)
        self._pending[job] = _Pending(mask, self._epoch)
        self.mask.clear()
        return job

    def finish(self, job: Job) -> JobOutcome:
        """מטפל ב-Job שהסתיים: הצלחה = Commit; כשל/ביטול ישיר = המסכה חוזרת לשכבה (PARITY-SPEC §3.6)."""
        pend = self._pending.get(job)
        if pend is None or pend.epoch != self._epoch:
            self._pending.pop(job, None)
            return JobOutcome(OutcomeKind.STALE)
        if not job.finished:
            raise InvalidInputError("העבודה טרם הסתיימה")
        del self._pending[job]
        if job.state is JobState.CANCELLED:
            self.mask.add(pend.mask)
            return JobOutcome(OutcomeKind.CANCELLED)
        if job.state is JobState.FAILED:
            self.mask.add(pend.mask)
            return JobOutcome(OutcomeKind.FAILED, error=job.error)
        try:
            patch = self.commit(job.result())
        except FixPicError as e:
            self.mask.add(pend.mask)
            return JobOutcome(OutcomeKind.FAILED, error=e)
        return JobOutcome(OutcomeKind.COMMITTED, patch=patch)

    def commit(self, result: InpaintResult) -> Patch | None:
        """Commit אטומי של תוצאת Inpaint: מדביק את פיקסלי המסכה האפקטיבית (ואת ה-Alpha, ראו `alpha.py`) לתמונה
        העובדת, ושומר Patch. מחזיר את ה-Patch, או None אם לא השתנה אף פיקסל. כשל = FixPicError והכול כפי שהיה."""
        h, w = self._rgb.shape[:2]
        em, img = result.effective_mask, result.image
        if (not isinstance(em, np.ndarray) or em.shape != (h, w) or not isinstance(img, np.ndarray)
                or img.shape != self._rgb.shape or img.dtype != np.uint8):
            raise InvalidInputError("תוצאת ה-Inpaint אינה בגודל התמונה העובדת")
        x, y, bw, bh = cv2.boundingRect(em)
        if bw == 0 or bh == 0:
            raise EmptyMaskError("המסכה האפקטיבית ריקה")
        box = (x, y, x + bw, y + bh)
        try:
            sub = em[y:y + bh, x:x + bw] > 0
            before = self._rgb[y:y + bh, x:x + bw][sub]
            after = img[y:y + bh, x:x + bw][sub]
            a_before = a_after = None
            if self._alpha is not None:
                a_before = self._alpha[y:y + bh, x:x + bw][sub]
                a_after = fill_alpha(self._alpha, em, box, sub)
            changed = bool((before != after).any()) or (a_before is not None and bool((a_before != a_after).any()))
            if not changed:
                return None
            patch = Patch.create(self.history.next_id(), box, (h, w), sub, before, after, a_before, a_after)
            new_rgb, new_alpha = patch.apply(self._rgb, self._alpha, forward=True)
        except MemoryError as e:
            raise OutOfMemoryError(f"נגמר הזיכרון ב-Commit: {e}") from e
        self._swap(new_rgb, new_alpha)
        self.history.push(patch)
        return patch

    # ------------------------------------------------------------- Undo / Redo
    def undo(self) -> HistoryAction:
        """Undo. בזמן עיבוד: מבטל את כל ה-Jobs (ההיסטוריה לא זזה); אחרת מבטל את ה-Commit האחרון."""
        if self._pending or (self.scheduler is not None and self.scheduler.current is not None):
            self._epoch += 1
            self._pending.clear()
            if self.scheduler is not None:
                self.scheduler.cancel_all()
            return HistoryAction.CANCELLED_JOBS
        patch = self.history.peek_undo()
        if patch is None:
            return HistoryAction.NOTHING
        self._apply_history(patch, forward=False)
        self.history.mark_undone()
        return HistoryAction.UNDONE

    def redo(self) -> HistoryAction:
        if self._pending:
            return HistoryAction.BUSY
        patch = self.history.peek_redo()
        if patch is None:
            return HistoryAction.NOTHING
        self._apply_history(patch, forward=True)
        self.history.mark_redone()
        return HistoryAction.REDONE

    # ----------------------------------------------------------------- פנימי
    def _apply_history(self, patch: Patch, *, forward: bool) -> None:
        try:
            new_rgb, new_alpha = patch.apply(self._rgb, self._alpha, forward=forward)
        except MemoryError as e:
            raise OutOfMemoryError(f"נגמר הזיכרון: {e}") from e
        self._epoch += 1
        self._swap(new_rgb, new_alpha)

    def _swap(self, rgb: np.ndarray, alpha: np.ndarray | None) -> None:
        self._rgb, self._alpha = _frozen(rgb), _frozen(alpha)
        self.version += 1
