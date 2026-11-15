"""בקר הבחירה החכמה (M4, AUTO-SELECT.md §7 שלב 4ב: חיווט GrabCut ל-Qt): מלבן / לאסו / משיכה מתקנת -> הצעה -> אישור.

`SmartSelectController` מחבר בין קלט הקנבס (`Canvas.selectBoxDrawn` ודומיו), `SelectionService` (Worker יחיד, `GrabCutProvider`) ושכבת
המסכה של ה-`Session`. הוא אינו מצייר ואינו מציג דיאלוגים: הוא משדר אותות והחלון הראשי מגיב.

חוזים (לפי AUTO-SELECT.md §4):
- **הצעה אינה נוגעת ב-`MaskLayer`, ב-`MaskHistory` או ב-`Session`.** רק `accept()` הוא עריכת מסכה: `Proposal.add_to` בתוך
  `MaskHistory.do`, כלומר צעד Undo אחד (Undo של לוח כלי המסכה), ואינו מפעיל Inpaint.
- **קלט חדש מחליף:** כל בקשה מבטלת את הקודמת (`SelectionService.propose`); `cancel()` מבטל בקשה רצה וזורק את ההצעה והקלט.
- **משיכה מתקנת** (`request_scribble`): נקודות חיוביות ("כן, זה חלק מהאובייקט") או שליליות ("לא") מתווספות לקלט הקיים (המלבן או
  הלאסו) ו-GrabCut רץ מחדש. הנקודות נבנות ב-`select.stroke.with_stroke` (דגימה צפופה לפי שטח המברשת; נקודה בודדת אינה משנה דבר,
  נמדד). המצב המצטבר נשמר עד אישור/ביטול/תמונה חדשה.
- **כשל בבקשה** מחזיר את הקלט וההצעה למצבם הקודם (משיכה שגויה לא הורסת הצעה טובה). **הצעה ריקה** (לא נמצא אובייקט) מוחקת את
  ההצעה אך משאירה את הקלט, כדי שמשיכה מתקנת נוספת תוכל לתקן.
- **Context של GrabCut תלוי בצבעים** (`stale_ok=False`), והמפתח הוא `(id(session), history.current_id)`: אחרי Commit/Undo/Redo
  נבנה Context חדש, וגם ההצעה הפתוחה נזרקת (`on_image_changed`), כי היא חושבה על תמונה שכבר אינה העובדת.
- GrabCut `heavy`: ה-Service ממתין כשיש Inpaint פעיל (`busy`), ובכל זאת ניתן לביטול. מחוון "הבחירה לוקחת זמן" רק אחרי 200ms.
- ה-callback של ה-Service רץ ב-thread של ה-Worker ורק משדר אות (Queued) ל-thread של ה-UI. `Session` נגיש רק מה-UI;
  `Session.rgb` אינו משתנה במקום (כל Commit מחליף מערך), ולכן בטוח להעביר אותו ל-Worker.
- **לא נבדק:** Windows, אירועי עכבר/טאבלט אמיתיים, i5-6500T (ראו PROGRESS).
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from ..errors import FixPicError
from ..select import (GrabCutProvider, Proposal, Prompts, SelectionError, SelectionService,
                      with_stroke)
from ..select.grabcut import MAX_POLYGON_POINTS, MIN_SIDE_PX
from ..select.service import RequestState
from ..session import Session
from ..session.mask_history import MaskHistory
from ..strings import t

log = logging.getLogger("fixpic.ui.select")

INDICATOR_DELAY_MS = 200      # כמו מחוון ה-Inpaint (PARITY-SPEC §3.5)


class SmartSelectController(QObject):
    changed = Signal()                  # הצעה/קלט/עיבוד השתנו: לצייר מחדש ולעדכן כפתורים
    message = Signal(str)               # הודעת סטטוס
    processingChanged = Signal(bool)    # מחוון "הבחירה לוקחת זמן" (אחרי 200ms)
    accepted = Signal(object)           # Rect | None: המלבן ששונה בשכבת המסכה
    _done = Signal(object)              # פנימי: מ-thread של Worker ל-UI

    def __init__(self, parent: QObject | None = None, *, provider_factory: Callable = GrabCutProvider,
                 history_source: Callable[[], MaskHistory | None] | None = None,
                 busy: Callable[[], bool] | None = None) -> None:
        """`history_source`: מחזיר את `MaskHistory` של שכבת המסכה (של לוח כלי המסכה), כדי שהאישור יהיה צעד Undo אחד שם.
        `busy`: האם Inpaint פעיל/ממתין (נקרא מה-Worker; קריאה בלבד)."""
        super().__init__(parent)
        self.session: Session | None = None
        self.proposal: Proposal | None = None
        self.indicator_on = False
        self._provider_factory = provider_factory
        self._history_source = history_source
        self._busy_probe = busy
        self._service: SelectionService | None = None
        self._prompts: Prompts | None = None        # הקלט המצטבר: מלבן או לאסו, ובנוסף נקודות המשיכות המתקנות
        self._req = None
        self._rollback: tuple[Prompts | None, Proposal | None] = (None, None)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(INDICATOR_DELAY_MS)
        self._timer.timeout.connect(self._show_indicator)
        self._done.connect(self._handle_done, Qt.QueuedConnection)

    # ------------------------------------------------------------------ מצב
    @property
    def has_proposal(self) -> bool:
        return self.proposal is not None

    @property
    def has_input(self) -> bool:
        """יש קלט פעיל (מלבן/לאסו), גם אם ההצעה ריקה או עדיין מחושבת: משיכה מתקנת אפשרית."""
        return self._prompts is not None

    @property
    def processing(self) -> bool:
        return self._req is not None and not self._req.finished

    @property
    def active(self) -> bool:
        """יש מה לבטל: הצעה, קלט או בקשה רצה."""
        return self.has_input or self.has_proposal or self.processing

    # ----------------------------------------------------------------- חיבור
    def bind(self, session: Session | None) -> None:
        """תמונה חדשה (או סגירה): כל מצב הבחירה נזרק, וה-Contexts של התמונה הקודמת משתחררים."""
        self._drop(silent=True)
        self.session = session
        if self._service is not None:
            self._service.invalidate()
        self.changed.emit()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._drop(silent=True)
        self._timer.stop()
        if self._service is not None:
            self._service.shutdown(wait=True, timeout=timeout)
            self._service = None

    def on_image_changed(self) -> None:
        """התמונה העובדת השתנתה (Commit/Undo/Redo): ההצעה חושבה על מצב ישן ונזרקת."""
        if self.active:
            self._drop(silent=True)
            self.changed.emit()

    # ------------------------------------------------------------------ בקשות
    def request_box(self, rect: Sequence[float]) -> bool:
        """מלבן גס סביב האובייקט, בפיקסלי תמונה (x0, y0, x1, y1; נחתך לתמונה ומעוגל החוצה)."""
        s = self.session
        if s is None:
            return False
        w, h = s.size
        try:
            x0, y0, x1, y1 = (float(v) for v in rect)
        except (TypeError, ValueError):
            return False
        if not all(map(math.isfinite, (x0, y0, x1, y1))):
            return False
        box = (max(0, math.floor(min(x0, x1))), max(0, math.floor(min(y0, y1))),
               min(w, math.ceil(max(x0, x1))), min(h, math.ceil(max(y0, y1))))
        if box[2] - box[0] < MIN_SIDE_PX or box[3] - box[1] < MIN_SIDE_PX:
            self.message.emit(t("select.too_small"))
            return False
        return self._submit(Prompts(box=box))

    def request_lasso(self, points: Sequence[tuple[float, float]]) -> bool:
        """לאסו גס (מצולע סגור אוטומטית), בפיקסלי תמונה."""
        s = self.session
        if s is None:
            return False
        w, h = s.size
        pts = [(min(max(float(x), 0.0), float(w)), min(max(float(y), 0.0), float(h))) for x, y in points
               if math.isfinite(x) and math.isfinite(y)]
        if len(pts) > MAX_POLYGON_POINTS:
            step = math.ceil(len(pts) / MAX_POLYGON_POINTS)
            pts = pts[::step]
        if len(pts) < 3:
            self.message.emit(t("select.too_small"))
            return False
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if max(xs) - min(xs) < MIN_SIDE_PX or max(ys) - min(ys) < MIN_SIDE_PX:
            self.message.emit(t("select.too_small"))
            return False
        return self._submit(Prompts(polygon=tuple(pts)))

    def request_scribble(self, path: Sequence[tuple[float, float]], positive: bool, brush_radius: float) -> bool:
        """משיכה מתקנת על הקלט הקיים: `positive=True` מוסיפה לאובייקט, False מחסירה. False אם אין קלט פעיל."""
        s = self.session
        if s is None or self._prompts is None:
            return False
        w, h = s.size
        try:
            new = with_stroke(self._prompts, path, brush_radius=max(1.0, float(brush_radius)), positive=positive,
                              bounds=(w, h))
        except SelectionError as e:
            self.message.emit(str(e))
            return False
        if len(new.points) == len(self._prompts.points):
            self.message.emit(t("select.scribble_outside"))
            return False
        return self._submit(new)

    def accept(self) -> tuple[int, int, int, int] | None:
        """מאשר את ההצעה לשכבת המסכה (צעד Undo אחד). מחזיר את המלבן ששונה, או None (אין הצעה / לא השתנה דבר)."""
        s, prop = self.session, self.proposal
        if s is None or prop is None:
            return None
        history = self._history_source() if self._history_source is not None else None
        try:
            if history is not None:
                rect = history.do(lambda layer: prop.add_to(layer))
            else:
                rect = prop.add_to(s.mask)
        except FixPicError as e:
            self.message.emit(str(e))
            return None
        self._drop(silent=True)
        self.message.emit(t("select.accepted") if rect is not None else t("select.nothing_new"))
        self.changed.emit()
        if rect is not None:
            self.accepted.emit(rect)
        return rect

    def cancel(self) -> bool:
        """מבטל בקשה רצה וזורק את ההצעה והקלט. True אם היה מה לבטל."""
        if not self.active:
            return False
        self._drop(silent=False)
        self.changed.emit()
        return True

    # ----------------------------------------------------------------- פנימי
    def _ensure_service(self) -> SelectionService:
        if self._service is None:
            self._service = SelectionService(self._provider_factory(), busy=self._busy_probe).start()
        return self._service

    def _submit(self, prompts: Prompts) -> bool:
        s = self.session
        if s is None:
            return False
        try:
            svc = self._ensure_service()
            key = (id(s), s.history.current_id)
            self._rollback = (self._prompts, self.proposal)
            self._prompts = prompts
            self._req = svc.propose(s.rgb, key, prompts, lambda r: self._done.emit(r))
        except FixPicError as e:
            self._prompts, self.proposal = self._rollback
            self.message.emit(str(e))
            self.changed.emit()
            return False
        self._timer.start()
        self.message.emit(t("select.working_short"))
        self.changed.emit()
        return True

    def _drop(self, *, silent: bool) -> None:
        req, self._req = self._req, None
        if req is not None and not req.finished:
            req.cancel()
        self._timer.stop()
        self._set_indicator(False)
        was = self.proposal is not None or self._prompts is not None
        self.proposal = None
        self._prompts = None
        self._rollback = (None, None)
        if was and not silent:
            self.message.emit(t("select.cancelled"))

    @Slot(object)
    def _handle_done(self, req) -> None:
        if req is not self._req:
            return                                  # בקשה שהוחלפה/בוטלה: מתעלמים
        self._req = None
        self._timer.stop()
        self._set_indicator(False)
        state = req.state
        if state is RequestState.CANCELLED:
            return
        if state is RequestState.DONE and req.proposal is not None:
            self.proposal = req.proposal
            self.message.emit(t("select.proposal", count=self.proposal.pixel_count))
        elif state is RequestState.EMPTY or (state is RequestState.DONE and req.proposal is None):
            self.proposal = None                    # הקלט נשאר: משיכה מתקנת אחרת אפשרית
            self.message.emit(t("select.empty"))
        else:                                       # FAILED: חוזרים למצב הקודם
            self._prompts, self.proposal = self._rollback
            err = req.error
            log.warning("בחירה חכמה נכשלה: %s", err)
            self.message.emit(t("select.failed", detail=str(err) if err is not None else ""))
        self._rollback = (None, None)
        self.changed.emit()

    def _show_indicator(self) -> None:
        if self.processing:
            self._set_indicator(True)

    def _set_indicator(self, on: bool) -> None:
        if on != self.indicator_on:
            self.indicator_on = on
            self.processingChanged.emit(on)
