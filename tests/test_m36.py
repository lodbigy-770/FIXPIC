"""M3.6: סגירת פערי P0 בממשק: רענון אזורי של הקנבס, ייצוא עם איכות, גלילה (Shift/Alt/אופקית) ונתיבים עבריים מקצה לקצה.
Qt offscreen, Scheduler אמיתי, מודל מדומה.
לא נבדק: Windows, מקלדת עברית, Big LaMa אמיתי, מראה בעין, i5-6500T."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.imgio import image_from_array, load_image  # noqa: E402
from fixpic.ui.canvas import PAN_STEP, Pyramid  # noqa: E402
from test_engine import rnd  # noqa: E402
from test_m34 import (EWin, app, drag, wait_until, ew, T, _close_canvases)  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("_close_canvases")


def full_pyr(a):
    return Pyramid(np.ascontiguousarray(a))


# ------------------------------------------------------------------ Pyramid.update_region
@pytest.mark.parametrize("hw", [(1500, 1100), (1501, 1101), (1301, 1700)])
def test_region_update_equals_full_rebuild(hw):
    h, w = hw
    base = rnd(h, w, 1)
    pyr = full_pyr(base)
    new = base.copy()
    box = (301, 205, 641, 513)
    new[box[1]:box[3], box[0]:box[2]] = rnd(box[3] - box[1], box[2] - box[0], 2)
    pyr.update_region(new, box)
    ref = full_pyr(new)
    assert len(pyr.arrays) == len(ref.arrays)
    for a, b in zip(pyr.arrays, ref.arrays):
        assert a.shape == b.shape and np.array_equal(a, b)


def test_region_update_at_edges_and_size_mismatch():
    base = rnd(1201, 1301, 3)
    pyr = full_pyr(base)
    new = base.copy()
    new[-7:, -9:] = 0
    new[:3, :5] = 255
    pyr.update_region(new, (1292, 1194, 1301, 1201))
    pyr.update_region(new, (0, 0, 5, 3))
    ref = full_pyr(new)
    assert all(np.array_equal(a, b) for a, b in zip(pyr.arrays, ref.arrays))
    with pytest.raises(ValueError):
        pyr.update_region(rnd(10, 10), (0, 0, 5, 5))


# ------------------------------------------------------------------ רענון אזורי של הקנבס (Alpha)
def test_canvas_region_refresh_with_alpha_matches_full_set(ew):
    h, w = 1200, 1500
    arr = np.dstack([rnd(h, w, 4), np.full((h, w), 255, np.uint8)])
    arr[::7, :, 3] = 120
    ew.w._install(image_from_array(arr), "a.png")
    ew.w.brush_panel.set_diameter(40)
    QApplication.processEvents()
    s, cv = ew.s, ew.canvas
    drag(cv, [(300, 300), (340, 300)])
    assert ew.idle() and s.history.undo_count == 1
    p0 = cv.pyr.arrays[0].copy()
    cv.set_image(s.rgb, s.alpha, keep_view=True)       # בנייה מלאה כהשוואה
    assert np.array_equal(p0, cv.pyr.arrays[0])
    assert all(np.array_equal(a, b) for a, b in zip(full_pyr(p0).arrays, cv.pyr.arrays))


def test_commit_composites_only_the_box_on_alpha_image(ew, monkeypatch):
    import fixpic.ui.canvas as cm
    h, w = 800, 1000
    arr = np.dstack([rnd(h, w, 5), np.full((h, w), 255, np.uint8)])
    arr[::5, :, 3] = 150                                  # Alpha מלא-אטום בלבד נזרק בטעינה
    ew.w._install(image_from_array(arr), "a.png")
    assert ew.s.alpha is not None
    ew.w.brush_panel.set_diameter(30)
    QApplication.processEvents()
    seen = []
    orig = cm.composite_on
    monkeypatch.setattr(cm, "composite_on", lambda rgb, a, bg: (seen.append(rgb.shape[:2]), orig(rgb, a, bg))[1])
    drag(ew.canvas, [(100, 100), (130, 100)])
    assert ew.idle() and ew.s.history.undo_count == 1
    assert seen and all(hh * ww < h * w // 4 for hh, ww in seen)


# ------------------------------------------------------------------ ייצוא עם איכות
def _setup_save(ew, tmp_path, name):
    ew.w._ask_discard = lambda: "discard"
    return tmp_path / name


def test_save_as_jpg_asks_quality_and_uses_it(ew, tmp_path):
    ew.install(h=200, w=300)
    asked = []
    ew.w._ask_quality = lambda fmt, cur: (asked.append((fmt, cur)), 55)[1]
    q55 = tmp_path / "q55.jpg"
    ew.w._pick_save = lambda sug, sel: (str(q55), sel)
    ew.w.save_as()
    assert wait_until(lambda: not ew.w.busy and q55.exists())
    assert asked == [("JPG", 92)] and ew.w.save_quality == 55
    ew.w._ask_quality = lambda fmt, cur: 95
    q95 = tmp_path / "q95.jpg"
    ew.w._pick_save = lambda sug, sel: (str(q95), sel)
    ew.w.save_as()
    assert wait_until(lambda: not ew.w.busy and q95.exists())
    assert q55.stat().st_size < q95.stat().st_size


def test_save_as_quality_cancel_aborts_and_png_does_not_ask(ew, tmp_path):
    ew.install(h=120, w=160)
    ew.w._ask_quality = lambda fmt, cur: None
    p = tmp_path / "x.webp"
    ew.w._pick_save = lambda sug, sel: (str(p), sel)
    ew.w.save_as()
    QApplication.processEvents()
    assert not p.exists() and not ew.w.busy
    called = []
    ew.w._ask_quality = lambda fmt, cur: called.append(fmt) or cur
    png = tmp_path / "x.png"
    ew.w._pick_save = lambda sug, sel: (str(png), sel)
    ew.w.save_as()
    assert wait_until(lambda: png.exists() and not ew.w.busy) and not called


# ------------------------------------------------------------------ גלילה
def wheel(cv, dx=0, dy=0, mods=Qt.NoModifier):
    pos = QPointF(cv.width() / 2, cv.height() / 2)
    ev = QWheelEvent(pos, cv.mapToGlobal(pos), QPoint(0, 0), QPoint(dx, dy), Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(cv, ev)


def test_wheel_zoom_plain_and_pan_with_modifiers(ew):
    ew.install(h=1500, w=2000)
    cv = ew.canvas
    cv.scale, cv.offset, cv._fit_mode = 1.0, QPointF(0, 0), False
    s0 = cv.scale
    wheel(cv, dy=120)
    assert cv.scale > s0                                   # גלגלת רגילה = זום
    cv.scale, cv.offset = 1.0, QPointF(0, 0)
    wheel(cv, dy=120, mods=Qt.ShiftModifier)
    assert cv.scale == 1.0 and cv.offset.x() != 0 and cv.offset.y() == 0
    dx = cv.offset.x()
    cv.offset = QPointF(0, 0)
    wheel(cv, dy=120, mods=Qt.AltModifier)
    assert cv.scale == 1.0 and cv.offset.x() == 0 and cv.offset.y() != 0
    cv.offset = QPointF(0, 0)
    wheel(cv, dx=120)
    assert cv.scale == 1.0 and cv.offset.x() == dx and cv.offset.y() == 0
    assert abs(dx) <= PAN_STEP + 1e-6 or abs(dx) == PAN_STEP


# ------------------------------------------------------------------ נתיבים עבריים מקצה לקצה
def test_hebrew_paths_open_stroke_save_ctrl_s_reload(ew, tmp_path):
    d = tmp_path / "תיקייה עברית" / "תמונות"
    d.mkdir(parents=True)
    src = d / "תמונה ראשונה.png"
    from fixpic.imgio import save_image
    save_image(src, rnd(240, 320, 7))
    w = ew.w
    w.open_path(src)
    assert wait_until(lambda: ew.s is not None and not w.busy)
    w.brush_panel.set_diameter(20)
    drag(ew.canvas, [(100, 100), (140, 100)])
    assert ew.idle() and ew.s.modified
    out = d / "תוצאה סופית.png"
    w._pick_save = lambda sug, sel: (str(out), sel)
    w.save_as()
    assert wait_until(lambda: out.exists() and not w.busy)
    assert not ew.s.modified
    saved = load_image(out).rgb
    assert np.array_equal(saved, ew.s.rgb)
    drag(ew.canvas, [(200, 150), (230, 150)])           # שינוי נוסף ואז Ctrl+S לאותו נתיב
    assert ew.idle() and ew.s.modified
    w.act_save.trigger()
    assert wait_until(lambda: not w.busy and not ew.s.modified)
    assert np.array_equal(load_image(out).rgb, ew.s.rgb)
    w.open_path(out)                                    # טעינה מחדש מתיקייה עברית
    assert wait_until(lambda: ew.s is not None and not w.busy and not ew.s.modified)
    assert np.array_equal(ew.s.rgb, load_image(out).rgb)
    assert "תוצאה סופית" in w.windowTitle()


# ------------------------------------------------------------------ תיקונים אחרי בדיקה ידנית (Windows)
def test_alt_wheel_with_swapped_axes_pans_vertically(ew):
    ew.install(h=1500, w=2000)
    cv = ew.canvas
    cv.scale, cv.offset, cv._fit_mode = 1.0, QPointF(0, 0), False
    wheel(cv, dx=120, mods=Qt.AltModifier)                 # Windows: Alt מחליף את הצירים, הדלתה מגיעה ב-x
    assert cv.offset.x() == 0 and cv.offset.y() != 0 and cv.scale == 1.0


def test_hebrew_letter_shortcuts_registered(ew):
    from PySide6.QtGui import QKeySequence
    w = ew.w
    for act, he in ((w.act_compare, "ב"), (w.act_tool_brush, "פ"), (w.act_tool_box, "ד"), (w.act_tool_lasso, "ך")):
        assert QKeySequence(he) in act.shortcuts()


def test_hebrew_hold_key_compares(ew):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    ew.install(h=200, w=300)
    for typ in (QEvent.KeyPress, QEvent.KeyRelease):
        QApplication.sendEvent(ew.canvas, QKeyEvent(typ, 0x05E0, Qt.NoModifier, "נ"))
        assert ew.w._cmp_hold == (typ == QEvent.KeyPress)


def test_accepting_a_selection_submits_it_to_inpaint(ew):
    """אישור הצעת בחירה חכמה = הסרה (בדיקה ידנית ב-Windows: בחירה בלי הסרה בלבלה)."""
    assert ew.ready()
    s = ew.install(h=200, w=300)
    s.mask.paint([(100.0, 100.0), (140.0, 100.0)], 30)
    assert not s.mask.empty
    ew.w._selection_accepted(None)
    assert ew.idle() and s.history.undo_count == 1 and s.mask.empty
