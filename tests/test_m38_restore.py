"""M3.8 (ב): מברשת שחזור, החזרת אזור מהתמונה המקורית כצעד Undo אחד."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from fixpic.errors import EmptyMaskError, InvalidInputError
from fixpic.imgio import image_from_array
from fixpic.session import Session
from test_m32 import fake_result, rnd


def make(alpha=False, size=(60, 80)):
    li = image_from_array(rnd(*size, seed=1))
    if alpha:
        li.alpha = np.random.default_rng(2).integers(50, 255, size, dtype=np.uint8)
    return Session(li)


def edit(s, box, value=7):
    """מדמה Commit קודם: צובע את החלון בערך קבוע דרך Patch אמיתי."""
    x0, y0, x1, y1 = box
    em = np.zeros(s.rgb.shape[:2], np.uint8)
    em[y0:y1, x0:x1] = 1
    return s.commit(fake_result(s.rgb, em, (value,) * 3))


def test_restore_returns_original_in_masked_area_only():
    s = make()
    orig = s.original_rgb.copy()
    edit(s, (10, 10, 50, 40))
    s.mask.paint([(30, 25)], 12)               # מעגל בתוך האזור שנערך
    patch = s.restore_from_original(feather_px=0)
    assert patch is not None and s.mask.empty
    yy, xx = np.mgrid[:60, :80]
    inside = (yy - 25) ** 2 + (xx - 30) ** 2 <= 5 ** 2
    assert (s.rgb[inside] == orig[inside]).all()
    assert (s.rgb[10:12, 10:12] == 7).all()    # שאר האזור שנערך נשאר


def test_restore_is_single_undo_step_and_redo():
    s = make()
    edit(s, (10, 10, 50, 40))
    before = s.rgb.copy()
    s.mask.paint([(30, 25)], 14)
    s.restore_from_original(feather_px=0)
    after = s.rgb.copy()
    assert not np.array_equal(before, after)
    s.undo()
    assert np.array_equal(s.rgb, before)
    s.redo()
    assert np.array_equal(s.rgb, after)


def test_restore_outside_edited_area_changes_nothing():
    s = make()
    edit(s, (10, 10, 20, 20))
    s.mask.paint([(60, 45)], 10)
    n = s.history.undo_count
    assert s.restore_from_original() is None and s.mask.empty
    assert s.history.undo_count == n


def test_feather_blends_edge_and_core_is_exact():
    s = make()
    orig = s.original_rgb.copy()
    edit(s, (0, 0, 80, 60), value=0)
    s.mask.paint([(40, 30)], 20)
    s.restore_from_original(feather_px=4)
    assert (s.rgb[30, 40] == orig[30, 40]).all()                      # מרכז = מקור מלא
    edge = s.rgb[30, 40 + 10 + 2].astype(int)                         # מעבר: בין 0 למקור
    assert (edge <= orig[30, 52]).all()


def test_restore_with_alpha():
    s = make(alpha=True)
    oa = s.original_alpha.copy()
    edit(s, (10, 10, 50, 40))
    s.mask.paint([(30, 25)], 14)
    s.restore_from_original(feather_px=0)
    assert s.alpha is not None and s.alpha[25, 30] == oa[25, 30]


def test_errors():
    s = make()
    with pytest.raises(EmptyMaskError):
        s.restore_from_original()
    s.mask.paint([(30, 25)], 10)
    s._pending[object()] = object()            # מדמה עיבוד פעיל
    with pytest.raises(InvalidInputError):
        s.restore_from_original()


# ---------------------------------------------------------------------------------- ממשק
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF  # noqa: E402

from test_m33 import app  # noqa: E402,F401
from fixpic.ui.edit_loop import EditController  # noqa: E402


def test_controller_restore_mode_commits_without_engine(app):
    s = make()
    edit(s, (10, 10, 50, 40))
    c = EditController()
    c.bind(s)
    c.restore_mode = True
    c.diameter = 12.0
    got = []
    c.imageChanged.connect(got.append)
    before = s.rgb.copy()
    assert c.begin_stroke(QPointF(30, 25))
    c.end_stroke()
    assert got and s.mask.empty and not np.array_equal(before, s.rgb)
    assert s.history.undo_count == 2           # העריכה + השחזור


def test_controller_restore_busy_discards_mask(app):
    s = make()
    edit(s, (10, 10, 50, 40))
    c = EditController()
    c.bind(s)
    c.restore_mode = True
    c._waiting = "engine"
    msgs = []
    c.message.connect(msgs.append)
    c.begin_stroke(QPointF(30, 25))
    c.end_stroke()
    assert s.mask.empty and msgs


def test_panel_restore_and_erase_exclusive_and_window_wiring(app):
    from fixpic.ui.main_window import MainWindow
    w = MainWindow()
    bp = w.brush_panel
    bp.set_erase(True)
    bp.set_restore(True)
    assert bp.restore and not bp.erase and w.controller.restore_mode and w.canvas.restore_mode
    bp.set_erase(True)
    assert bp.erase and not bp.restore and not w.controller.restore_mode
