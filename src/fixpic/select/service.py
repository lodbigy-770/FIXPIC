"""`SelectionService` (AUTO-SELECT.md §5): Worker יחיד לבחירה, cache של Contexts, ביטול, והמתנה ל-Inpaint.

חוזים:
- ה-callbacks (`on_done`) רצים ב-thread של ה-Worker (או של המבטל, לבקשה שעמדה בתור). ב-Qt: להעביר ל-signal.
- `propose` חדש מבטל כל `propose` קודם (ממתין או רץ) וכל Context-רקע ("קלט חדש מבטל הצעה רצה"). בקשה שבוטלה לא
  מחזירה הצעה, גם אם החישוב הסתיים בפועל. `warm` חדש מבטל `warm` קודם.
- `key`: מזהה גרסת תמונה (למשל `(מזהה Session, history.current_id)`); Context נשמר לפיו. cache ב-LRU, עד
  `max_contexts` (ברירת מחדל 2). `invalidate()` משחרר (סגירת תמונה).
- Provider עם `stale_ok=True`: אם אין Context למפתח, משתמשים בחדש ביותר מאותו גודל תמונה (`request.stale=True`) ומכינים
  Context מדויק ברקע. Provider עם `stale_ok=False` תמיד מקבל Context מדויק. **[מדיניות ה-stale לא נבדקה עם מודל]**
- Provider עם `heavy=True` ממתין כש-`busy()` מחזיר True (Inpaint פעיל), ובכל זאת ניתן לביטול. ל-`heavy=False` אין המתנה.
- חריגה לא צפויה מה-Provider נעטפת ב-`SelectionError` ונרשמת ביומן.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import OrderedDict, deque
from enum import Enum
from typing import Callable, Hashable

import numpy as np

from ..errors import CancelledError, FixPicError, SchedulerClosedError
from ..models.base import CancelToken
from .base import SelectionProvider
from .types import Context, Prompts, Proposal, SelectionError

log = logging.getLogger("fixpic.select")

DEFAULT_MAX_CONTEXTS = 2


class RequestState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"            # propose: יש הצעה; warm: ה-Context מוכן
    EMPTY = "empty"          # propose: הבחירה ריקה (לא שגיאה)
    FAILED = "failed"
    CANCELLED = "cancelled"


_FINAL = (RequestState.DONE, RequestState.EMPTY, RequestState.FAILED, RequestState.CANCELLED)


class SelectionRequest:
    """בקשה אחת. נוצרת רק ע"י `SelectionService`."""

    def __init__(self, service: "SelectionService", req_id: int, kind: str, rgb: np.ndarray, key: Hashable,
                 prompts: Prompts | None, on_done, background: bool = False) -> None:
        self._svc = service
        self.id, self.kind, self.key, self.background = req_id, kind, key, background
        self.prompts = prompts
        self.token = CancelToken()
        self.state = RequestState.QUEUED
        self.stale = False                       # נענתה מ-Context של גרסה קודמת
        self._rgb = rgb                          # משוחרר בסיום
        self._proposal: Proposal | None = None
        self._error: BaseException | None = None
        self._on_done = on_done
        self._event = threading.Event()

    def cancel(self) -> bool:
        return self._svc.cancel(self)

    @property
    def finished(self) -> bool:
        return self._event.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def proposal(self) -> Proposal | None:
        return self._proposal

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def result(self, timeout: float | None = None) -> Proposal | None:
        """ההצעה (None: בחירה ריקה, או `warm`). `TimeoutError` אם לא הסתיימה, `CancelledError` אם בוטלה, ואת
        שגיאת הבקשה אם נכשלה."""
        if not self._event.wait(timeout):
            raise TimeoutError(f"בקשה {self.id} לא הסתיימה")
        if self.state is RequestState.CANCELLED:
            raise CancelledError("הפעולה בוטלה")
        if self._error is not None:
            raise self._error
        return self._proposal

    def _finish(self, state: RequestState, proposal: Proposal | None = None, error: BaseException | None = None) -> None:
        self.state, self._proposal, self._error = state, proposal, error
        self._rgb = None
        self._event.set()
        if self._on_done:
            try:
                self._on_done(self)
            except Exception:  # noqa: BLE001
                log.exception("on_done נכשל (בקשה %d)", self.id)


class SelectionService:
    def __init__(self, provider: SelectionProvider, *, busy: Callable[[], bool] | None = None,
                 max_contexts: int = DEFAULT_MAX_CONTEXTS, busy_poll: float = 0.05) -> None:
        if max_contexts < 1:
            raise ValueError("max_contexts חייב להיות לפחות 1")
        self.provider = provider
        self._busy = busy
        self._max_contexts = max_contexts
        self._busy_poll = busy_poll
        self._cond = threading.Condition()
        self._queue: deque[SelectionRequest] = deque()
        self._current: SelectionRequest | None = None
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._stop = False
        self._cache_lock = threading.Lock()
        self._cache: "OrderedDict[Hashable, Context]" = OrderedDict()

    # -------------------------------------------------------------- מחזור חיים
    def start(self) -> "SelectionService":
        with self._cond:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="fixpic-select", daemon=True)
                self._thread.start()
        return self

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        self._cancel_where(lambda r: True)
        with self._cond:
            self._stop = True
            self._cond.notify_all()
            t = self._thread
        if wait and t is not None and t is not threading.current_thread():
            t.join(timeout)
        self.invalidate()

    def __enter__(self) -> "SelectionService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # -------------------------------------------------------------------- API
    def warm(self, rgb: np.ndarray, key: Hashable, on_done=None) -> SelectionRequest:
        """מכין Context לגרסת התמונה ברקע (למשל בטעינה ואחרי Commit). מבטל `warm` קודם."""
        return self._submit("warm", rgb, key, None, on_done, cancel_kinds=("warm",))

    def propose(self, rgb: np.ndarray, key: Hashable, prompts: Prompts, on_done=None) -> SelectionRequest:
        """מבקש הצעה. מבטל `propose` קודם ו-Context-רקע. `rgb` אינו משתנה עד סיום הבקשה."""
        if not isinstance(prompts, Prompts):
            raise SelectionError("prompts חייב להיות Prompts")
        return self._submit("propose", rgb, key, prompts, on_done, cancel_kinds=("propose", "background"))

    def cancel(self, req: SelectionRequest) -> bool:
        return bool(self._cancel_where(lambda r: r is req))

    def invalidate(self, key: Hashable | None = None) -> None:
        """משחרר Context של מפתח (או את כולם)."""
        with self._cache_lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    @property
    def context_keys(self) -> list:
        with self._cache_lock:
            return list(self._cache)

    # ------------------------------------------------------------------ פנימי
    def _submit(self, kind, rgb, key, prompts, on_done, cancel_kinds, background=False) -> SelectionRequest:
        if not (isinstance(rgb, np.ndarray) and rgb.ndim == 3):
            raise SelectionError("התמונה חייבת להיות מערך [H,W,3]")
        with self._cond:
            if self._thread is None or self._stop:
                raise SchedulerClosedError("שירות הבחירה לא רץ")
        self._cancel_where(lambda r: ("background" in cancel_kinds and r.background)
                           or (r.kind in cancel_kinds and not r.background))
        with self._cond:
            if self._thread is None or self._stop:
                raise SchedulerClosedError("שירות הבחירה לא רץ")
            req = SelectionRequest(self, next(self._ids), kind, rgb, key, prompts, on_done, background)
            self._queue.append(req)
            self._cond.notify_all()
            return req

    def _cancel_where(self, pred) -> int:
        removed: list[SelectionRequest] = []
        n = 0
        with self._cond:
            for r in list(self._queue):
                if pred(r):
                    self._queue.remove(r)
                    removed.append(r)
                    n += 1
            cur = self._current
            if cur is not None and cur.state not in _FINAL and pred(cur):
                cur.token.cancel()
                n += 1
        for r in removed:                       # מחוץ לנעילה: ה-callback עלול לקרוא ל-Service
            r.token.cancel()
            r._finish(RequestState.CANCELLED)
        return n

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._stop and not self._queue:
                    self._cond.wait()
                if self._stop:
                    return
                req = self._queue.popleft()
                req.state = RequestState.RUNNING
                self._current = req
            try:
                self._run(req)
            except BaseException as e:  # noqa: BLE001  רשת ביטחון: ה-Worker לא מת
                log.exception("כשל לא צפוי בבקשה %d", req.id)
                if not req.finished:
                    req._finish(RequestState.FAILED, error=SelectionError(str(e)))
            finally:
                with self._cond:
                    self._current = None

    def _run(self, req: SelectionRequest) -> None:
        try:
            self._wait_not_busy(req)
            req.token.raise_if_cancelled()
            ctx = self._context(req)
            if req.kind == "warm":
                req._finish(RequestState.DONE)
                return
            prop = self.provider.propose(ctx, req.prompts, req.token)
            req.token.raise_if_cancelled()          # בוטלה: לא מחזירים הצעה
            if prop is not None and prop.window.shape[::-1] != (prop.rect[2] - prop.rect[0], prop.rect[3] - prop.rect[1]):
                raise SelectionError("הצעה לא תקינה מה-Provider")
            req._finish(RequestState.EMPTY if prop is None else RequestState.DONE, prop)
        except CancelledError:
            req._finish(RequestState.CANCELLED)
        except FixPicError as e:
            req._finish(RequestState.FAILED, error=e)
        except Exception as e:  # noqa: BLE001
            log.exception("Provider נכשל (בקשה %d)", req.id)
            req._finish(RequestState.FAILED, error=SelectionError(f"הבחירה נכשלה: {e}"))

    def _wait_not_busy(self, req: SelectionRequest) -> None:
        if not (self.provider.heavy and self._busy):
            return
        while self._busy():
            req.token.raise_if_cancelled()
            time.sleep(self._busy_poll)

    def _context(self, req: SelectionRequest) -> Context:
        shape = req._rgb.shape[:2]
        with self._cache_lock:
            ctx = self._cache.get(req.key)
            if ctx is not None:
                self._cache.move_to_end(req.key)
                return ctx
            stale = None
            if req.kind == "propose" and self.provider.stale_ok:
                for k in reversed(self._cache):
                    if self._cache[k].shape == shape:
                        stale = self._cache[k]
                        break
        if stale is not None:
            req.stale = True
            self._queue_background_prepare(req)
            return stale
        ctx = self.provider.prepare(req._rgb, req.token)
        req.token.raise_if_cancelled()
        with self._cache_lock:
            self._cache[req.key] = ctx
            self._cache.move_to_end(req.key)
            while len(self._cache) > self._max_contexts:
                self._cache.popitem(last=False)
        return ctx

    def _queue_background_prepare(self, req: SelectionRequest) -> None:
        """אחרי מענה מ-Context ישן: הכנת Context מדויק ברקע (בקשה שקופה, נעצרת בקלט הבא)."""
        try:
            self._submit("warm", req._rgb, req.key, None, None, cancel_kinds=("background",), background=True)
        except FixPicError:
            pass
