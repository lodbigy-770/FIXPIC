"""M3.10: התאמה למסכי 14 אינץ' - לוחות בגלילה, גודל חלון לפי המסך, וקיצורי מקלדת להסרת רקע ושקיפות."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.ui.scroll import MIN_WINDOW, PREFERRED_WINDOW, PanelScroll, initial_window_size  # noqa: E402
from test_m33 import app  # noqa: E402,F401
from test_m4_select_ui import SWin, drag  # noqa: E402


def test_initial_window_size_fits_small_screens():
    assert initial_window_size(None) == PREFERRED_WINDOW
    assert initial_window_size(QSize(3840, 2160)) == PREFERRED_WINDOW
    s = initial_window_size(QSize(1366, 700))                     # 14" טיפוסי אחרי סרגל משימות
    assert s.width() <= 1366 and s.height() <= 700 * 0.92 + 1
    tiny = initial_window_size(QSize(300, 200))
    assert tiny.width() >= MIN_WINDOW.width() and tiny.height() >= MIN_WINDOW.height()


def test_docks_are_scrollable_and_panels_still_reachable(app):
    sw = SWin()
    try:
        w = sw.w
        for dock, panel in ((w.brush_dock, w.brush_panel), (w.mask_dock, w.mask_panel), (w.select_dock, w.select_panel)):
            sc = dock.widget()
            assert isinstance(sc, PanelScroll) and sc.panel is panel and sc.widgetResizable()
        assert w.minimumSize() == MIN_WINDOW
    finally:
        sw.close()


def test_small_window_scrolls_panels_instead_of_growing(app):
    sw = SWin()
    try:
        w = sw.w
        w.show()
        w.mask_dock.setVisible(True)                                  # הכי צפוף: שלושת הלוחות יחד
        w.resize(1366, 680)                                           # 14" אחרי סרגל משימות
        QApplication.processEvents()
        assert w.height() == 680 and w.minimumSizeHint().height() <= MIN_WINDOW.height() + 20
        bars = [d.widget().verticalScrollBar().maximum() for d in (w.brush_dock, w.mask_dock, w.select_dock)]
        assert max(bars) > 0                                          # לפחות לוח אחד נגלל
        for d in (w.brush_dock, w.mask_dock, w.select_dock):
            assert d.widget().horizontalScrollBar().maximum() == 0     # בלי גלילה אופקית
        sc = w.select_dock.widget()                                   # הגלילה מגיעה לכפתור האחרון
        sc.verticalScrollBar().setValue(sc.verticalScrollBar().maximum())
        QApplication.processEvents()
        assert sc.panel.btn_cancel.mapTo(sc.viewport(), sc.panel.btn_cancel.rect().center()).y() <= sc.viewport().height()
    finally:
        sw.close()


def test_bg_and_transparent_shortcuts_defined_with_hebrew_letters(app):
    sw = SWin()
    try:
        w = sw.w
        assert QKeySequence("G") in w.act_remove_bg.shortcuts() and QKeySequence("ע") in w.act_remove_bg.shortcuts()
        assert QKeySequence("T") in w.act_make_transparent.shortcuts()
        assert QKeySequence("א") in w.act_make_transparent.shortcuts()
        assert not w.act_remove_bg.isEnabled() and not w.act_make_transparent.isEnabled()    # אין הצעה
        w.act_make_transparent.trigger()                                                    # לא קורס ולא משנה
        assert sw.s.alpha is None
    finally:
        sw.close()


@pytest.mark.parametrize("which", ["bg", "tr"])
def test_shortcut_actions_apply_proposal(app, which):
    sw = SWin()
    try:
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        act = w.act_remove_bg if which == "bg" else w.act_make_transparent
        assert act.isEnabled()
        act.trigger()
        QApplication.processEvents()
        a = sw.s.alpha
        assert a is not None and (a == 0).mean() > 0.05 and (a == 255).mean() > 0.05
        assert not w.select_ctl.has_proposal and sw.s.history.undo_count == 1
        assert not act.isEnabled()
        if which == "bg":
            assert a[100, 100] == 255                                   # האובייקט נשאר, הקצה שקוף
        else:
            assert a[100, 100] == 0
    finally:
        sw.close()
