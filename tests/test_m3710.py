"""M3.7.10: התקדמות רב-שלבית והודעת עיבוד ארוך בממשק (Qt offscreen, Scheduler אמיתי, מודל מדומה).

לא נבדק כאן: מראה בעין, Windows, זמן אמיתי של Big LaMa (הסף LONG_DELAY_MS מוקטן בבדיקות)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from fixpic import strings  # noqa: E402
from fixpic.ui import edit_loop  # noqa: E402
from test_m34 import T, app, drag, ew, ew_gate, wait_until  # noqa: E402,F401


def test_strings_present():
    assert strings.t("status.processing_stage", step=2, total=5) == "שלב 2 מתוך 5"
    assert "ארוך" in strings.t("status.processing_long")
    assert edit_loop.LONG_DELAY_MS > edit_loop.INDICATOR_DELAY_MS


def test_long_notice_after_delay_and_cleared_on_finish(ew_gate):
    ew = ew_gate
    ew.install()
    ew.c._long_timer.setInterval(300)
    longs = []
    ew.c.longProcessingChanged.connect(longs.append)
    drag(ew.canvas, [(100, 100)])
    assert ew.model.entered.acquire(timeout=T)
    assert wait_until(lambda: ew.c.indicator_on, 3)
    assert longs == [] and "ארוך" not in ew.w.lbl_proc.text()          # המחוון כבר דולק, ההודעה עוד לא
    assert wait_until(lambda: longs == [True], 3)
    assert "ארוך" in ew.w.lbl_proc.text() and "מעבד" in ew.w.lbl_proc.text()
    ew.model.gate.release(1)
    assert ew.idle()
    assert longs == [True, False] and not ew.c.long_on
    assert ew.w.lbl_proc.text() == "" and ew.w.proc_progress.isHidden()


def test_fast_job_never_shows_long_notice(ew):
    ew.install(h=120, w=160)
    longs = []
    ew.c.longProcessingChanged.connect(longs.append)
    drag(ew.canvas, [(60, 60), (80, 60)])
    assert ew.idle() and longs == [] and not ew.c.long_on


def test_stage_text_for_multistage_and_not_for_single(ew_gate):
    ew = ew_gate
    ew.install()
    drag(ew.canvas, [(100, 100)])
    assert ew.model.entered.acquire(timeout=T)
    assert wait_until(lambda: ew.c.indicator_on, 3)
    ew.w._processing_progress(0, 1)                                     # חלון יחיד: בלי "שלב"
    assert "שלב" not in ew.w.lbl_proc.text()
    ew.w._processing_progress(2, 5)
    assert "שלב 3 מתוך 5" in ew.w.lbl_proc.text()
    assert ew.w.proc_progress.maximum() == 5 and ew.w.proc_progress.value() == 2
    ew.w._processing_progress(5, 5)                                     # הושלם: בלי טקסט שלב
    assert "שלב" not in ew.w.lbl_proc.text()
    ew.model.gate.release(1)
    assert ew.idle()
    assert ew.w.lbl_proc.text() == "" and ew.w._proc_stage == (0, 0)


def test_stage_and_long_texts_combine(ew_gate):
    ew = ew_gate
    ew.install()
    ew.c._long_timer.setInterval(300)
    drag(ew.canvas, [(100, 100)])
    assert ew.model.entered.acquire(timeout=T)
    assert wait_until(lambda: ew.c.long_on, 3)
    ew.w._processing_progress(1, 3)
    txt = ew.w.lbl_proc.text()
    assert "שלב 2 מתוך 3" in txt and "ארוך" in txt
