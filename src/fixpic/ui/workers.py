"""הרצת פעולות כבדות (טעינה/שמירה) מחוץ ל-thread של ה-UI (DEVELOPMENT_LOG M3.1: שמירה של 24MP לוקחת שניות).

`BackgroundRunner.submit(fn, on_done, on_failed)`: `fn` רצה ב-QThreadPool; התוצאה/החריגה מוחזרות ל-thread של ה-UI
דרך signal אל slot של אובייקט שחי שם, ולכן `on_done`/`on_failed` רצים תמיד ב-thread של ה-UI. `fn` אסור שתיגע ב-Qt
או ב-Session (חד-thread): היא מקבלת רק נתונים בלתי-משתנים (מערכי התמונה הם קריאה-בלבד)."""
from __future__ import annotations

import itertools
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, object)


class _Task(QRunnable):
    def __init__(self, token: int, fn: Callable, signals: _Signals) -> None:
        super().__init__()
        self.token, self.fn, self.signals = token, fn, signals

    def run(self) -> None:
        try:
            result = self.fn()
        except BaseException as e:  # noqa: BLE001 - כל חריגה חוזרת ל-UI; ה-thread לא מת
            self.signals.failed.emit(self.token, e)
        else:
            self.signals.done.emit(self.token, result)


class BackgroundRunner(QObject):
    def __init__(self, parent: QObject | None = None, pool: QThreadPool | None = None) -> None:
        super().__init__(parent)
        self._pool = pool or QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._signals = _Signals(self)
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)
        self._ids = itertools.count(1)
        self._active: dict[int, tuple[Callable, Callable]] = {}

    @property
    def active(self) -> int:
        return len(self._active)

    def submit(self, fn: Callable, on_done: Callable, on_failed: Callable) -> int:
        token = next(self._ids)
        self._active[token] = (on_done, on_failed)
        self._pool.start(_Task(token, fn, self._signals))
        return token

    def wait_idle(self, timeout_ms: int = 30000) -> bool:
        return self._pool.waitForDone(timeout_ms)

    @Slot(int, object)
    def _on_done(self, token: int, result) -> None:
        cb = self._active.pop(token, None)
        if cb:
            cb[0](result)

    @Slot(int, object)
    def _on_failed(self, token: int, error) -> None:
        cb = self._active.pop(token, None)
        if cb:
            cb[1](error)
