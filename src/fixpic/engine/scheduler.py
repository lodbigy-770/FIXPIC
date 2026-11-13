"""Job Scheduler (ARCHITECTURE §4א, ADR-009): תור FIFO של עד 3 עבודות ממתינות, Worker יחיד שמריץ עבודה אחת
בכל זמן, ו-Warm-up של ה-backend ברקע (compile לכל bucket, הגודל הרגיל קודם).

חוזים:
- ה-callbacks (`on_progress`, `on_done`) רצים ב-thread של ה-Worker. ממשק משתמש חייב להעביר אותם ל-thread שלו
  (ב-Qt: signal). חריגה ב-callback נרשמת ביומן ואינה משפיעה על העבודה.
- ה-Scheduler אינו מעתיק את התמונה והמסכה: המתקשר לא משנה אותן עד סיום העבודה (Commit יוצר מערך חדש).
- עבודה שבוטלה אינה מחזירה תוצאה, גם אם ההרצה הסתיימה בפועל לפני שהביטול נבדק: Undo לא יכול להיכתב.
- עבודה נכשלת לעולם לא משנה את התמונה (Pipeline.run מחזיר מערך חדש).
- ה-Worker אינו נקטע באמצע הרצת מודל (אינה ניתנת להפסקה); ביטול נבדק בין חלונות.
- Warm-up מתבצע bucket אחד בכל פעם, ועבודה שממתינה בתור קודמת ל-bucket הבא. עבודה שמגיעה באמצע compile
  ממתינה לסיומו (עד כ-6 שניות על i5-6500T, ערך M1.5). buckets שלא נקמפלו יתקמפלו בשימוש ראשון."""
from __future__ import annotations

import itertools
import logging
import threading
from collections import deque
from enum import Enum
from typing import Callable

import numpy as np

from ..errors import (CancelledError, FixPicError, InferenceError, QueueFullError, SchedulerClosedError)
from ..models.base import CancelToken
from ..pipeline import InpaintResult, Pipeline, Progress

log = logging.getLogger("fixpic.scheduler")

MAX_PENDING = 3   # PARITY-SPEC §3: עד 3 ממתינות (לא כולל העבודה הרצה)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WarmupState(str, Enum):
    IDLE = "idle"            # טרם התבקש
    PENDING = "pending"      # ממתין/רץ
    DONE = "done"
    FAILED = "failed"        # הפרטים ב-warmup_error; העבודות ממשיכות לעבוד (compile בשימוש ראשון)
    CANCELLED = "cancelled"  # נעצר בסגירה


_FINAL = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


class Job:
    """עבודה אחת. נוצרת רק ע\"י JobScheduler.submit."""

    def __init__(self, scheduler: "JobScheduler", job_id: int, image: np.ndarray, mask: np.ndarray,
                 size: int | None, on_progress, on_done,
                 brush_px: float | None = None) -> None:
        self._sched = scheduler
        self.id = job_id
        self.size = size
        self.brush_px = brush_px
        self.token = CancelToken()
        self.state = JobState.QUEUED
        self._image, self._mask = image, mask      # משוחררים בסיום (24MP = כ-72MB לתמונה)
        self._result: InpaintResult | None = None
        self._error: BaseException | None = None
        self._on_progress, self._on_done = on_progress, on_done
        self._event = threading.Event()

    # ------------------------------------------------------------ ממשק למתקשר
    def cancel(self) -> bool:
        """מבטל. ממתינה: מוסרת מהתור מיד. רצה: ביטול נבדק בין חלונות. True אם הביטול נרשם (העבודה לא הסתיימה)."""
        return self._sched.cancel(self)

    @property
    def finished(self) -> bool:
        return self._event.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def wait(self, timeout: float | None = None) -> bool:
        """ממתין לסיום (כל מצב סופי). True אם הסתיימה בזמן."""
        return self._event.wait(timeout)

    def result(self, timeout: float | None = None) -> InpaintResult:
        """התוצאה; מחכה עד timeout. זורק TimeoutError אם לא הסתיימה, CancelledError אם בוטלה, ואת שגיאת העבודה אם נכשלה."""
        if not self._event.wait(timeout):
            raise TimeoutError(f"עבודה {self.id} לא הסתיימה")
        if self.state is JobState.CANCELLED:
            raise CancelledError("הפעולה בוטלה")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    # ------------------------------------------------------------ פנימי (Worker)
    def _emit_progress(self, p: Progress) -> None:
        if self._on_progress:
            try:
                self._on_progress(self, p)
            except Exception:  # noqa: BLE001
                log.exception("on_progress נכשל (עבודה %d)", self.id)

    def _finish(self, state: JobState, result=None, error=None) -> None:
        self.state, self._result, self._error = state, result, error
        self._image = self._mask = None
        self._event.set()
        if self._on_done:
            try:
                self._on_done(self)
            except Exception:  # noqa: BLE001
                log.exception("on_done נכשל (עבודה %d)", self.id)


class JobScheduler:
    def __init__(self, pipeline: Pipeline, max_pending: int = MAX_PENDING) -> None:
        if max_pending < 1:
            raise ValueError("max_pending חייב להיות לפחות 1")
        self.pipeline = pipeline
        self.max_pending = max_pending
        self.warmup_state = WarmupState.IDLE
        self.warmup_error: BaseException | None = None
        self._cond = threading.Condition()
        self._queue: deque[Job] = deque()
        self._current: Job | None = None
        self._warm_todo: list[int] = []
        self._warm_cancel = CancelToken()
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._stop = False

    # -------------------------------------------------------------- מחזור חיים
    def start(self, warmup: bool = True) -> "JobScheduler":
        """מפעיל את ה-Worker ואת ה-Warm-up ברקע. קריאה שנייה היא no-op."""
        with self._cond:
            if self._thread is not None:
                return self
            self._thread = threading.Thread(target=self._loop, name="fixpic-worker", daemon=True)
            if warmup:
                self._warm_todo = self.pipeline.warmup_order()
                self.warmup_state = WarmupState.PENDING
            self._thread.start()
            self._cond.notify_all()
        return self

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """מבטל את כל העבודות ועוצר. הרצת מודל שכבר התחילה מסתיימת לפני העצירה. המודל עצמו לא נפרק."""
        self.cancel_all()
        with self._cond:
            self._stop = True
            self._warm_todo.clear()
            self._warm_cancel.cancel()
            if self.warmup_state is WarmupState.PENDING:
                self.warmup_state = WarmupState.CANCELLED
            self._cond.notify_all()
            t = self._thread
        if wait and t is not None and t is not threading.current_thread():
            t.join(timeout)

    def __enter__(self) -> "JobScheduler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # -------------------------------------------------------------------- API
    def submit(self, image: np.ndarray, mask: np.ndarray, size: int | None = None,
               on_progress: Callable[[Job, Progress], None] | None = None,
               on_done: Callable[[Job], None] | None = None, brush_px: float | None = None) -> Job:
        """brush_px: קוטר המברשת בפיקסלי התמונה (להרחבת מסכה אוטומטית; None = הערכה מהמסכה). מוסיף עבודה לסוף התור. QueueFullError אם יש כבר max_pending ממתינות; SchedulerClosedError אם לא רץ."""
        with self._cond:
            if self._thread is None or self._stop:
                raise SchedulerClosedError("ה-Scheduler לא רץ")
            if len(self._queue) >= self.max_pending:
                raise QueueFullError(f"התור מלא ({self.max_pending} ממתינות)")
            job = Job(self, next(self._ids), image, mask, size, on_progress, on_done, brush_px)
            self._queue.append(job)
            self._cond.notify_all()
            return job

    def cancel(self, job: Job) -> bool:
        removed = False
        with self._cond:
            if job.state in _FINAL:
                return False
            if job.state is JobState.QUEUED:
                self._queue.remove(job)
                removed = True
            job.token.cancel()          # ל-RUNNING: ה-Worker יסיים כ-CANCELLED
        if removed:
            job._finish(JobState.CANCELLED)   # מחוץ לנעילה: callback עלול לקרוא ל-Scheduler
        return True

    def cancel_all(self) -> int:
        """Undo בזמן עיבוד (ARCHITECTURE §4א, Race): מבטל את הרצה ומרוקן את התור. מחזיר כמה עבודות בוטלו."""
        with self._cond:
            dropped = list(self._queue)
            self._queue.clear()
            running = self._current
            for j in dropped:
                j.token.cancel()
            if running is not None:
                running.token.cancel()
        for j in dropped:
            j._finish(JobState.CANCELLED)
        return len(dropped) + (1 if running is not None else 0)

    @property
    def pending(self) -> int:
        with self._cond:
            return len(self._queue)

    @property
    def current(self) -> Job | None:
        with self._cond:
            return self._current

    def wait_warmup(self, timeout: float | None = None) -> bool:
        """ממתין לסיום ה-Warm-up (הצלחה, כשל או ביטול). לבדיקות ולכלים; ה-UI אינו צריך לחכות."""
        with self._cond:
            return self._cond.wait_for(lambda: self.warmup_state is not WarmupState.PENDING, timeout)

    # ----------------------------------------------------------------- Worker
    def _loop(self) -> None:
        while True:
            job, bucket = None, None
            with self._cond:
                while True:
                    if self._stop:
                        return
                    if self._queue:                       # עבודה קודמת ל-Warm-up
                        job = self._queue.popleft()
                        job.state = JobState.RUNNING
                        self._current = job
                        break
                    if self._warm_todo:
                        bucket = self._warm_todo.pop(0)
                        break
                    self._cond.wait()
            if job is not None:
                self._execute(job)
                with self._cond:
                    self._current = None
            else:
                self._warm_one(bucket)

    def _execute(self, job: Job) -> None:
        try:
            res = self.pipeline.run(job._image, job._mask, job.size, cancel=job.token,
                                    progress=job._emit_progress, brush_px=job.brush_px)
        except CancelledError:
            job._finish(JobState.CANCELLED)
        except FixPicError as e:
            log.warning("עבודה %d נכשלה: %s", job.id, e)
            job._finish(JobState.FAILED, error=e)
        except Exception as e:  # noqa: BLE001 - ה-Worker לעולם לא מת מחריגה (ARCHITECTURE §10)
            log.exception("עבודה %d: חריגה לא צפויה", job.id)
            err = InferenceError(f"שגיאה לא צפויה: {e}")
            err.__cause__ = e
            job._finish(JobState.FAILED, error=err)
        else:
            if job.token.cancelled:
                job._finish(JobState.CANCELLED)
            else:
                job._finish(JobState.DONE, result=res)

    def _warm_one(self, bucket: int) -> None:
        err: BaseException | None = None
        try:
            self.pipeline.warmup(self._warm_cancel, [bucket])
        except CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 - כשל Warm-up אינו חוסם עבודות (compile בשימוש ראשון)
            log.warning("Warm-up ל-bucket %d נכשל: %s", bucket, e)
            err = e
        with self._cond:
            if err is not None:
                self._warm_todo.clear()               # כשל ב-backend: אין טעם להמשיך ל-buckets הבאים
                self.warmup_error = err
                self.warmup_state = WarmupState.FAILED
            elif self._stop:
                self.warmup_state = WarmupState.CANCELLED
            elif not self._warm_todo:
                self.warmup_state = WarmupState.DONE
            self._cond.notify_all()
