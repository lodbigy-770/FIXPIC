"""M3.8: הסרת רקע מינימלית (GrabCut + Alpha): `Session.remove_background` והחיווט בממשק."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from fixpic.errors import EmptyMaskError, InvalidInputError
from fixpic.imgio import encode_image, image_from_array
from fixpic.session import Session
from test_m32 import fake_result, rnd


def make(alpha=False, size=(60, 80)):
    li = image_from_array(rnd(*size, seed=3))
    if alpha:
        li.alpha = np.random.default_rng(4).integers(100, 255, size, dtype=np.uint8)
    return Session(li)


def obj(x0=20, y0=15, x1=50, y1=45):
    return (x0, y0, x1, y1), np.full((y1 - y0, x1 - x0), 255, np.uint8)


def test_background_becomes_transparent_object_stays_opaque_rgb_untouched():
    s = make()
    rgb0 = s.rgb.copy()
    rect, win = obj()
    patch = s.remove_background(rect, win, feather_px=0)
    assert patch is not None and s.alpha is not None
    assert (s.alpha[15:45, 20:50] == 255).all()
    assert s.alpha[0, 0] == 0 and s.alpha[59, 79] == 0 and s.alpha[10, 40] == 0
    assert np.array_equal(s.rgb, rgb0)


def test_undo_redo_single_step_restores_alpha():
    s = make()
    rect, win = obj()
    s.remove_background(rect, win, feather_px=0)
    a1 = s.alpha.copy()
    s.undo()
    assert (s.alpha == 255).all()
    s.redo()
    assert np.array_equal(s.alpha, a1)


def test_undo_earlier_rgb_only_patch_after_alpha_created():
    s = make()
    em = np.zeros((60, 80), np.uint8)
    em[5:15, 5:15] = 1
    s.commit(fake_result(s.rgb, em, (9, 9, 9)))          # Patch בלי Alpha
    before = s.rgb.copy()
    rect, win = obj()
    s.remove_background(rect, win)
    s.undo()
    s.undo()                                              # Undo של ה-Patch הישן על תמונה שיש בה Alpha
    assert not np.array_equal(s.rgb, before)
    s.redo()
    s.redo()
    assert s.alpha[0, 0] == 0 and (s.rgb[5:15, 5:15] == 9).all()


def test_feather_makes_soft_edge():
    s = make()
    rect, win = obj()
    s.remove_background(rect, win, feather_px=3)
    edge = s.alpha[30, 18:24]
    assert 0 < edge.min() or (edge > 0).any()
    assert s.alpha[30, 35] == 255 and s.alpha[30, 5] == 0
    assert len(set(edge.tolist())) > 2                    # מעבר הדרגתי


def test_existing_alpha_is_multiplied_not_replaced():
    s = make(alpha=True)
    a0 = s.alpha.copy()
    rect, win = obj()
    s.remove_background(rect, win, feather_px=0)
    assert np.array_equal(s.alpha[20:40, 30:40], a0[20:40, 30:40])
    assert (s.alpha[:10] == 0).all()


def test_errors_and_noop():
    s = make()
    rect, win = obj()
    with pytest.raises(EmptyMaskError):
        s.remove_background(rect, np.zeros_like(win))
    with pytest.raises(InvalidInputError):
        s.remove_background((0, 0, 200, 200), np.zeros((200, 200), np.uint8))
    full = (0, 0, 80, 60), np.full((60, 80), 255, np.uint8)
    assert s.remove_background(*full, feather_px=0) is None      # אין רקע
    s._pending[object()] = object()
    with pytest.raises(InvalidInputError):
        s.remove_background(rect, win)


def test_restore_brush_gives_back_opaque_after_bg_removal():
    s = make()
    rect, win = obj()
    s.remove_background(rect, win, feather_px=0)
    s.mask.paint([(5, 5)], 6)
    s.restore_from_original(feather_px=0)
    assert s.alpha[5, 5] == 255


def test_png_keeps_transparency_jpg_flattens():
    s = make()
    rect, win = obj()
    s.remove_background(rect, win, feather_px=0)
    _, fl_png = encode_image(s.rgb, s.alpha, fmt="PNG")
    _, fl_jpg = encode_image(s.rgb, s.alpha, fmt="JPEG")
    assert not fl_png and fl_jpg


pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF  # noqa: E402,F401

from test_m33 import app  # noqa: E402,F401


from test_m4_select_ui import SWin, drag, iou, _full  # noqa: E402,F401
from fixpic.ui.select_panel import SelectPanel  # noqa: E402


def test_panel_button_state(app):
    p = SelectPanel()
    assert not p.btn_bg.isEnabled()
    p.set_state(True, True)
    assert p.btn_bg.isEnabled()


def test_end_to_end_box_then_remove_background_then_undo(app):
    from PySide6.QtWidgets import QApplication
    sw = SWin()
    try:
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        assert w.select_panel.btn_bg.isEnabled()
        w.select_panel.btn_bg.click()
        QApplication.processEvents()
        a = sw.s.alpha
        assert a is not None and (a == 0).mean() > 0.2 and (a == 255).mean() > 0.05
        assert not w.select_ctl.has_proposal                       # ההצעה נוקתה
        assert sw.s.history.undo_count == 1
        w.act_undo.trigger()
        QApplication.processEvents()
        assert (sw.s.alpha == 255).all()
    finally:
        sw.close()
