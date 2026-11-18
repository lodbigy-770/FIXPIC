"""M3.9: הפיכת נבחר לשקוף, מברשת שקיפות, והחלה ידנית."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from fixpic.errors import EmptyMaskError, InvalidInputError
from test_m38_bg import make, obj


def test_make_transparent_is_inverse_of_remove_background():
    s = make()
    rect, win = obj()
    rgb0 = s.rgb.copy()
    s.make_transparent(rect, win, feather_px=0)
    assert (s.alpha[15:45, 20:50] == 0).all()
    assert s.alpha[0, 0] == 255 and s.alpha[59, 79] == 255
    assert np.array_equal(s.rgb, rgb0)
    s.undo()
    assert (s.alpha == 255).all()


def test_transparent_then_keep_object_compose():
    s = make()
    s.make_transparent(*obj(20, 15, 40, 30), feather_px=0)           # חור
    s.remove_background(*obj(10, 10, 60, 50), feather_px=0)           # שמירת אזור גדול יותר
    assert s.alpha[20, 30] == 0 and s.alpha[40, 55] == 255 and s.alpha[2, 2] == 0


def test_transparent_brush_from_mask():
    s = make()
    s.mask.paint([(30, 25), (50, 25)], 10)
    patch = s.make_transparent_from_mask(feather_px=0)
    assert patch is not None and s.mask.empty
    assert s.alpha[25, 30] == 0 and s.alpha[25, 50] == 0 and s.alpha[5, 5] == 255
    s.undo()
    assert (s.alpha == 255).all()


def test_transparent_brush_errors_and_noop():
    s = make()
    with pytest.raises(EmptyMaskError):
        s.make_transparent_from_mask()
    s.mask.paint([(30, 25)], 8)
    assert s.make_transparent_from_mask(feather_px=0) is not None
    s.mask.paint([(30, 25)], 8)                                       # כבר שקוף
    assert s.make_transparent_from_mask(feather_px=0) is None and s.mask.empty
    s.mask.paint([(30, 25)], 8)
    s._pending[object()] = object()
    with pytest.raises(InvalidInputError):
        s.make_transparent_from_mask()


def test_restore_brush_undoes_transparency():
    s = make()
    s.make_transparent(*obj(), feather_px=0)
    s.mask.paint([(35, 30)], 10)
    s.restore_from_original(feather_px=0)
    assert s.alpha[30, 35] == 255 and s.alpha[16, 21] == 0


pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_m33 import app  # noqa: E402,F401
from test_m4_select_ui import SWin, drag  # noqa: E402
from fixpic.ui.brush_panel import BrushPanel  # noqa: E402


def test_brush_panel_exclusivity(app):
    p = BrushPanel()
    p.set_restore(True)
    p.set_transparent(True)
    assert p.transparent and not p.restore
    p.set_manual(True)
    assert p.manual and not p.transparent
    p.set_erase(True)                      # מחיקה מתאימה להחלה ידנית
    assert p.erase and p.manual
    p.set_restore(True)
    assert p.restore and not p.manual and not p.erase


def test_manual_apply_accumulates_and_button_state(app):
    sw = SWin()
    try:
        w, c = sw.w, sw.w.controller
        w.brush_panel.set_manual(True)
        assert c.manual_apply and not w.brush_panel.btn_apply.isEnabled()
        c.begin_stroke(QPointF(100, 100)); c.extend_stroke(QPointF(140, 100)); c.end_stroke()
        QApplication.processEvents()
        assert not sw.s.mask.empty and not sw.s.busy                # לא הוגש
        assert w.brush_panel.btn_apply.isEnabled()
        assert c.cancel() and sw.s.mask.empty                        # Esc זורק את המסכה שנצברה
    finally:
        sw.close()


def test_transparent_brush_in_window_and_pending_mask_dropped(app):
    sw = SWin()
    try:
        w, c = sw.w, sw.w.controller
        w.brush_panel.set_manual(True)
        c.begin_stroke(QPointF(100, 100)); c.end_stroke()
        assert not sw.s.mask.empty
        w.brush_panel.set_transparent(True)                          # המסכה הממתינה נזרקת
        assert sw.s.mask.empty and not c.manual_apply and c.transparent_mode
        c.diameter = 30.0
        c.begin_stroke(QPointF(100, 100)); c.end_stroke()
        QApplication.processEvents()
        assert sw.s.alpha is not None and sw.s.alpha[100, 100] == 0 and sw.s.alpha[5, 5] == 255
        w.act_undo.trigger()
        QApplication.processEvents()
        assert (sw.s.alpha == 255).all()
    finally:
        sw.close()


def test_make_selected_transparent_button_end_to_end(app):
    sw = SWin()
    try:
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done() and w.select_panel.btn_tr.isEnabled()
        w.select_panel.btn_tr.click()
        QApplication.processEvents()
        a = sw.s.alpha
        assert a is not None and (a == 0).mean() > 0.05 and (a == 255).mean() > 0.2
        assert not w.select_ctl.has_proposal and sw.s.history.undo_count == 1
    finally:
        sw.close()


def test_mask_tools_visible_only_in_manual_mode(app):
    sw = SWin()
    try:
        w = sw.w
        assert w.mask_dock.isHidden() and not w.select_panel.radios["mask_fill"].isVisibleTo(w.select_panel)
        assert not w.brush_panel.chk_erase.isVisibleTo(w.brush_panel)
        w.brush_panel.set_manual(True)
        assert not w.mask_dock.isHidden()
        assert w.select_panel.radios["mask_fill"].isVisibleTo(w.select_panel)
        assert w.brush_panel.chk_erase.isVisibleTo(w.brush_panel)
        w.act_tool_mask_fill.trigger()
        assert w.select_panel.tool == "mask_fill"
        w.brush_panel.set_manual(False)                      # חוזרים למברשת, ומצב מחיקה כבוי
        assert w.select_panel.tool == "brush" and w.mask_dock.isHidden() and not w.brush_panel.erase
    finally:
        sw.close()
