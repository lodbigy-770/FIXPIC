"""לולאת העריכה (M3.4, PARITY-SPEC §3): משיכת מברשת -> מסכה בשכבה -> שחרור = Job ב-Scheduler -> Commit.

`EditController` מחבר בין קלט המשיכות של ה-Canvas, `Session` (מסכה, Commit) ו-`JobScheduler` (עיבוד ברקע). הוא אינו
מצייר ואינו מציג דיאלוגים: הוא משדר אותות והחלון הראשי מגיב.

חוזים:
- **Thread אחד ל-Session.** ה-callbacks של ה-Scheduler רצים ב-thread של ה-Worker ורק משדרים אות (בחיבור Queued)
  ל-thread של ה-UI; שם נקרא `Session.finish`.
- **שחרור מברשת (ציור) = הגשה מיידית.** משיכת מחיקה (כפתור ימני או מצב מחיקה) רק מסירה מהמסכה ואינה מריצה דבר.
- **תור מלא** (3 ממתינות, §3.3): המסכה נשארת בשכבה ומוגשת אוטומטית כשעבודה מסתיימת. **מנוע בטעינה**: אותו דבר, ומוגשת
  כשהמנוע מוכן. כשל בעיבוד (§3.6): המסכה חוזרת לשכבה והתמונה לא משתנה, **ולא** מוגשת מחדש אוטומטית (אחרת לולאת כשלים).
- **מחוון עיבוד רק אחרי 200ms** (§3.5): `processingChanged(True)` נשלח אם העיבוד עדיין נמשך אז, ו-`False` בסיומו.
- **ביטול (`cancel`)**: מבטל את כל המשיכות שבעיבוד ובתור וזורק את מסכותיהן (`Session.cancel_pending`), בלי לגעת
  בהיסטוריה; מסכה שמחכה לתור/למנוע נזרקת גם היא.
- **Undo/Redo (M3.5):** `undo()` בזמן עיבוד או המתנה (תור/מנוע) = ביטול כל המשיכות (PARITY-SPEC §3.4 סעיף 4), בלי לבטל Commit;
  רק כשאין עבודה הוא מבטל Commit אחרון. `redo()` בזמן עיבוד נדחה. שניהם נחסמים באמצע משיכת מברשת. אחרי שינוי תמונה
  משודר `imageChanged` (אותו אות גם אחרי Commit), והחלון מרענן את הקנבס.
- מצב המנוע: `NONE` (לא הוגדר מנוע; משיכות רק מציירות מסכה, כמו בבדיקות), `LOADING`, `READY`, `FAILED`.
"""
from __future__ import annotations

import logging
from enum import Enum

from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal, Slot

from ..engine import Job, JobScheduler
from ..errors import FixPicError, QueueFullError, SchedulerClosedError
from ..pipeline import Progress
from ..session import HistoryAction, JobOutcome, OutcomeKind, Session
from ..strings import t

log = logging.getLogger("fixpic.ui.edit")

INDICATOR_DELAY_MS = 200      # PARITY-SPEC §3.5


class EngineState(str, Enum):
    NONE = "none"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class EditController(QObject):
    changed = Signal()                  # המסכה/העבודות השתנו: לצייר מחדש ולעדכן כפתורים
    committed = Signal(object)          # JobOutcome עם Patch: התמונה העובדת השתנתה
    imageChanged = Signal()             # התמונה העובדת השתנתה (Commit, Undo או Redo): לרענן את התצוגה
    message = Signal(str)               # הודעת סטטוס
    failed = Signal(str)                # כשל בעיבוד (טקסט מלא להצגה)
    processingChanged = Signal(bool)    # מחוון עיבוד (אחרי 200ms)
    progress = Signal(int, int)         # done, total של העבודה הרצה
    _job_done = Signal(object)          # פנימי: מ-thread של Worker ל-UI
    _job_progress = Signal(int, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.session: Session | None = None
        self.scheduler: JobScheduler | None = None
        self.state = EngineState.NONE
        self.engine_error: str | None = None
        self.diameter = 20.0                # קוטר המברשת בפיקסלי תמונה
        self.erase_mode = False
        self.indicator_on = False
        self._stroke: dict | None = None    # {"erase": bool, "last": (x, y)}
        self._waiting: str | None = None    # "queue" | "engine": מסכה בשכבה שמחכה להגשה
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(INDICATOR_DELAY_MS)
        self._timer.timeout.connect(self._show_indicator)
        self._job_done.connect(self._handle_done, Qt.QueuedConnection)
        self._job_progress.connect(self._handle_progress, Qt.QueuedConnection)

    # ----------------------------------------------------------------- חיבור
    def bind(self, session: Session | None) -> None:
        """מחבר Session (תמונה חדשה). עבודות של ה-Session הקודם מבוטלות (הן שייכות לתמונה אחרת)."""
        old = self.session
        if old is not None and old is not session:
            old.cancel_pending()
        self.session = session
        self._stroke = None
        self._waiting = None
        if session is not None and self.state is EngineState.READY:
            session.scheduler = self.scheduler
        self._sync_indicator()
        self.changed.emit()

    def engine_loading(self) -> None:
        self.state = EngineState.LOADING
        self.engine_error = None

    def engine_ready(self, scheduler: JobScheduler) -> None:
        self.scheduler, self.state, self.engine_error = scheduler, EngineState.READY, None
        if self.session is not None:
            self.session.scheduler = scheduler
        if self._waiting == "engine":
            self._waiting = None
            if self._stroke is None:
                self.submit()
        self.changed.emit()

    def engine_failed(self, detail: str) -> None:
        self.state, self.engine_error = EngineState.FAILED, detail
        if self._waiting == "engine" and self.session is not None:
            self._waiting = None
            self.session.mask.clear()
        self.changed.emit()

    def shutdown(self, timeout: float = 5.0) -> None:
        """סוגר את ה-Scheduler (עבודה שכבר רצה במודל מסתיימת לפני העצירה)."""
        if self.session is not None:
            self.session.cancel_pending()
        self._timer.stop()
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=True, timeout=timeout)
            self.scheduler = None
        if self.state is EngineState.READY:
            self.state = EngineState.NONE

    # ----------------------------------------------------------------- משיכות
    @property
    def stroking(self) -> bool:
        return self._stroke is not None

    @property
    def waiting(self) -> str | None:
        return self._waiting

    def begin_stroke(self, pt: QPointF, erase: bool = False) -> bool:
        """תחילת משיכה בנקודת תמונה (מרכז פיקסל). `erase` או `erase_mode` = מחיקה. False אם אין Session."""
        s = self.session
        if s is None:
            return False
        erase = bool(erase or self.erase_mode)
        xy = (pt.x(), pt.y())
        try:
            s.mask.paint([xy], self.diameter, erase=erase)
        except FixPicError as e:
            self.message.emit(str(e))
            return False
        self._stroke = {"erase": erase, "last": xy}
        self.changed.emit()
        return True

    def extend_stroke(self, pt: QPointF) -> None:
        st, s = self._stroke, self.session
        if st is None or s is None:
            return
        xy = (pt.x(), pt.y())
        try:
            s.mask.paint([st["last"], xy], self.diameter, erase=st["erase"])
        except FixPicError as e:
            self.message.emit(str(e))
            return
        st["last"] = xy
        self.changed.emit()

    def end_stroke(self) -> None:
        """שחרור המברשת: ציור = הגשת Job; מחיקה = רק עדכון תצוגה."""
        st, self._stroke = self._stroke, None
        if st is None:
            return
        if st["erase"]:
            self.changed.emit()
            return
        self.submit()

    # --------------------------------------------------------------- הגשה
    def submit(self) -> None:
        """מגיש את מסכת השכבה כ-Job (אם יש מסכה ומנוע). ראו חוזים למצבי תור מלא/מנוע בטעינה/כשל."""
        s = self.session
        if s is None or s.mask.empty:
            self.changed.emit()
            return
        if self.state is EngineState.NONE:
            self.changed.emit()                       # אין מנוע מוגדר: המסכה רק נשארת מצוירת
            return
        if self.state is EngineState.FAILED:
            s.mask.clear()
            self.message.emit(t("error.no_engine"))
            self.changed.emit()
            return
        if self.state is EngineState.LOADING or s.scheduler is None:
            self._waiting = "engine"
            self.message.emit(t("status.engine_wait"))
            self.changed.emit()
            return
        try:
            s.submit_stroke(on_progress=self._on_progress, on_done=self._on_done)
        except QueueFullError:
            self._waiting = "queue"
            self.message.emit(t("status.queue_full"))
            self.changed.emit()
            return
        except SchedulerClosedError as e:
            s.mask.clear()
            self.failed.emit(str(e))
            self.changed.emit()
            return
        except FixPicError as e:
            self.failed.emit(str(e))
            self.changed.emit()
            return
        self._waiting = None
        self._sync_indicator()
        self.changed.emit()

    def cancel(self) -> bool:
        """ביטול עיבוד (Esc). True אם בוטל משהו (עבודות, או מסכה שחיכתה להגשה)."""
        s = self.session
        if s is None:
            return False
        did = s.cancel_pending()
        if self._waiting is not None and not s.mask.empty:
            s.mask.clear()
            did = True
        self._waiting = None
        self._sync_indicator()
        if did:
            self.message.emit(t("status.job_cancelled"))
        self.changed.emit()
        return did

    # ------------------------------------------------------------ Undo / Redo
    @property
    def can_undo(self) -> bool:
        s = self.session
        return s is not None and not self.stroking and (s.history.can_undo or s.busy or self._waiting is not None)

    @property
    def can_redo(self) -> bool:
        s = self.session
        return s is not None and not self.stroking and not s.busy and s.history.can_redo

    def undo(self) -> HistoryAction:
        s = self.session
        if s is None or self.stroking:
            return HistoryAction.NOTHING
        if self._waiting is not None:                  # מסכה שמחכה לתור/מנוע: Undo מבטל אותה כמו עיבוד
            self.cancel()
            return HistoryAction.CANCELLED_JOBS
        try:
            act = s.undo()
        except FixPicError as e:
            self.failed.emit(str(e))
            return HistoryAction.NOTHING
        if act is HistoryAction.CANCELLED_JOBS:
            self._sync_indicator()
            self.message.emit(t("status.job_cancelled"))
            self.changed.emit()
        elif act is HistoryAction.UNDONE:
            self.message.emit(t("status.undone"))
            self.imageChanged.emit()
            self.changed.emit()
        return act

    def redo(self) -> HistoryAction:
        s = self.session
        if s is None or self.stroking:
            return HistoryAction.NOTHING
        try:
            act = s.redo()
        except FixPicError as e:
            self.failed.emit(str(e))
            return HistoryAction.NOTHING
        if act is HistoryAction.REDONE:
            self.message.emit(t("status.redone"))
            self.imageChanged.emit()
            self.changed.emit()
        elif act is HistoryAction.BUSY:
            self.message.emit(t("status.redo_busy"))
        return act

    # ------------------------------------------- callbacks (thread של ה-Worker)
    def _on_done(self, job: Job) -> None:
        self._job_done.emit(job)

    def _on_progress(self, _job: Job, p: Progress) -> None:
        self._job_progress.emit(p.done, p.total)

    # ----------------------------------------------------------- thread של ה-UI
    @Slot(int, int)
    def _handle_progress(self, done: int, total: int) -> None:
        self.progress.emit(done, total)

    @Slot(object)
    def _handle_done(self, job: Job) -> None:
        s = self.session
        if s is None:
            return
        try:
            outcome = s.finish(job)
        except FixPicError as e:
            log.error("finish נכשל", exc_info=e)
            self.failed.emit(str(e))
            outcome = None
        if outcome is not None:
            self._on_outcome(outcome)
        self._sync_indicator()
        if self._waiting == "queue" and self._stroke is None:
            self.submit()               # התפנה מקום בתור
            return
        self.changed.emit()

    def _on_outcome(self, o: JobOutcome) -> None:
        if o.kind is OutcomeKind.COMMITTED:
            if o.patch is None:
                self.message.emit(t("status.no_change"))
            else:
                self.committed.emit(o)
                self.imageChanged.emit()
        elif o.kind is OutcomeKind.FAILED:
            log.warning("עיבוד נכשל: %s", o.error)
            self.failed.emit(str(o.error) if o.error is not None else "")
        elif o.kind is OutcomeKind.CANCELLED:
            self.message.emit(t("status.job_cancelled"))

    # ------------------------------------------------------------------ מחוון
    def _sync_indicator(self) -> None:
        busy = self.session is not None and self.session.busy
        if busy:
            if not self.indicator_on and not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            if self.indicator_on:
                self.indicator_on = False
                self.processingChanged.emit(False)

    def _show_indicator(self) -> None:
        if self.session is not None and self.session.busy and not self.indicator_on:
            self.indicator_on = True
            self.processingChanged.emit(True)
