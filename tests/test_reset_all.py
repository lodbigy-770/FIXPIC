"""איפוס הכל: Session, EditController ו-MainWindow (Scheduler אמיתי, מודל מדומה, Qt offscreen).

לא נבדק: מראה בעין של הדיאלוג, Windows."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from fixpic.imgio import LoadedImage  # noqa: E402
from fixpic.session import Session  # noqa: E402
from test_m34 import T, app, drag, ew, ew_gate, wait_until  # noqa: E402,F401


def _edited(ew):
    ew.install()
    drag(ew.canvas, [(100, 100), (140, 100)])
    assert ew.idle() and ew.s.history.can_undo
    assert not np.array_equal(ew.s.rgb, ew.s.original_rgb)


def test_reset_restores_original_and_clears_everything(ew):
    _edited(ew)
    ew.w._ask_reset = lambda: True
    ew.w.brush_panel.set_diameter(77)
    ew.w.brush_panel.set_erase(True)
    ew.w.reset_all()
    s = ew.s
    assert np.array_equal(s.rgb, s.original_rgb) and s.mask.empty
    assert not s.history.can_undo and not s.history.can_redo and not ew.c.can_undo and not ew.w.act_undo.isEnabled()
    assert not s.modified
    assert not ew.w.brush_panel.erase and ew.w.brush_panel.spin.value() != 77 and ew.w.brush_panel.chk_feather_auto.isChecked()
    assert ew.w.select_panel.radios["brush"].isChecked()


def test_reset_declined_changes_nothing(ew):
    _edited(ew)
    before = ew.s.rgb.copy()
    ew.w._ask_reset = lambda: False
    ew.w.reset_all()
    assert np.array_equal(ew.s.rgb, before) and ew.s.history.can_undo and ew.s.modified


def test_no_confirmation_when_nothing_to_lose(ew):
    ew.install()
    ew.w._ask_reset = lambda: (_ for _ in ()).throw(AssertionError("אין מה לאבד: אסור לשאול"))
    ew.w.reset_all()
    assert ew.s.mask.empty and not ew.s.modified


def test_reset_is_enabled_with_image_and_in_menus_and_toolbar(ew):
    from PySide6.QtWidgets import QMenu, QToolBar
    assert not ew.w.act_reset.isEnabled()                      # בלי תמונה
    ew.install()
    assert ew.w.act_reset.isEnabled()
    assert ew.w.act_reset.text() == "איפוס הכל"
    assert ew.w.act_reset in [a for tb in ew.w.findChildren(QToolBar) for a in tb.actions()]
    assert ew.w.act_reset in [a for m in ew.w.menuBar().findChildren(QMenu) for a in m.actions()]


def test_reset_during_processing_cancels_job_and_never_commits_it(ew_gate):
    ew = ew_gate
    ew.install()
    ew.w._ask_reset = lambda: True
    drag(ew.canvas, [(100, 100)])
    assert ew.model.entered.acquire(timeout=T)
    assert ew.s.busy
    ew.w.reset_all()
    assert not ew.s.busy and ew.c.waiting is None
    ew.model.gate.release(5)
    assert ew.idle()
    ew.w._runner.wait_idle()
    assert np.array_equal(ew.s.rgb, ew.s.original_rgb) and not ew.s.history.can_undo


def test_reset_after_save_marks_modified_and_keeps_alpha():
    from fixpic.history import Patch
    rgb = np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    alpha = np.full((40, 50), 200, np.uint8)
    s = Session(LoadedImage(rgb=rgb, alpha=alpha, icc_profile=None, exif=None, source_path=None, source_format="png"))
    n = 40 * 50
    s._swap(np.zeros_like(rgb), np.full((40, 50), 7, np.uint8))        # מדמה Commit: תמונה שונה + Patch בהיסטוריה
    s.history.push(Patch.create(s.history.next_id(), (0, 0, 50, 40), (40, 50), np.ones((40, 50), bool),
                                rgb.reshape(n, 3).copy(), np.zeros((n, 3), np.uint8), alpha.reshape(n).copy(), np.full(n, 7, np.uint8)))
    s.mark_saved()
    assert not s.modified
    assert s.reset_all() is True
    assert np.array_equal(s.rgb, rgb) and np.array_equal(s.alpha, alpha)
    assert s.modified                                                 # שונה מהקובץ השמור
    assert s.reset_all() is False                                     # פעם שנייה: אין מה לאפס
