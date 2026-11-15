"""M4: לאסו ישיר על המסכה בממשק (מילוי ומחיקה), מעל `MaskLayer.fill_polygon`. Qt offscreen.

מוכיח: האות `maskLassoDrawn` מהקנבס (בלי לגעת במסכה ובלי לשדר בחירה חכמה), מילוי/מחיקה כצעד Undo אחד בלוח כלי המסכה,
הגשה ל-Inpaint אחרי מילוי (ולא אחרי מחיקה), התעלמות ממצולע מחוץ לתמונה, קיצורים והחלפת כלי.
**לא נבדק:** Windows, עכבר/טאבלט אמיתיים, מקלדת עברית (M/D לפי האות), מראה בעין."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from test_m34 import T, app, drag, painted_canvas, wait_until, ew, _close_canvases  # noqa: E402,F401
from test_m4_select_ui import SWin, sw  # noqa: E402,F401

pytestmark = pytest.mark.usefixtures("_close_canvases")

SQUARE = [(60, 50), (160, 50), (160, 130), (60, 130)]


class TestCanvas:
    @pytest.mark.parametrize("mode,erase", [("mask_fill", False), ("mask_erase", True)])
    def test_emits_mask_lasso_not_select_lasso_and_never_paints(self, app, mode, erase):
        c, layer = painted_canvas(h=200, w=300)
        c.set_select_mode(mode)
        mask_got, sel_got = [], []
        c.maskLassoDrawn.connect(lambda pts, e: mask_got.append((pts, e)))
        c.selectLassoDrawn.connect(sel_got.append)
        drag(c, [(50, 50), (150, 50), (150, 120), (50, 120)])
        assert len(mask_got) == 1 and not sel_got and layer.empty
        pts, e = mask_got[0]
        assert e is erase and len(pts) >= 4
        assert min(p[0] for p in pts) == pytest.approx(50.5, abs=1.0)

    def test_two_points_are_dropped(self, app):
        c, _layer = painted_canvas(h=200, w=300)
        c.set_select_mode("mask_fill")
        got = []
        c.maskLassoDrawn.connect(lambda pts, e: got.append(pts))
        drag(c, [(50, 50), (150, 80)])
        assert not got

    def test_plain_select_lasso_still_goes_to_selection(self, app):
        c, _layer = painted_canvas(h=200, w=300)
        c.set_select_mode("lasso")
        mask_got, sel_got = [], []
        c.maskLassoDrawn.connect(lambda pts, e: mask_got.append(pts))
        c.selectLassoDrawn.connect(sel_got.append)
        drag(c, [(50, 50), (150, 50), (150, 120), (50, 120)])
        assert len(sel_got) == 1 and not mask_got


class TestWindow:
    def test_fill_adds_polygon_to_mask_as_one_undo_step(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("mask_fill")
        assert cv.select_mode == "mask_fill"
        drag(cv, SQUARE)
        m = sw.s.mask.array
        assert m[90, 110] > 0 and m[10, 10] == 0
        assert sw.s.mask.bbox() is not None
        assert w.mask_panel.history.can_undo
        w.mask_panel.undo()
        assert sw.s.mask.empty
        w.mask_panel.redo()
        assert sw.s.mask.array[90, 110] > 0

    def test_erase_removes_polygon_from_existing_mask(self, sw):
        w, cv = sw.w, sw.canvas
        w.mask_panel.fill_polygon([(40, 30), (200, 30), (200, 150), (40, 150)])
        before = sw.s.mask.pixel_count
        w.select_panel.set_tool("mask_erase")
        drag(cv, [(80, 60), (140, 60), (140, 110), (80, 110)])
        assert 0 < sw.s.mask.pixel_count < before
        assert sw.s.mask.array[85, 110] == 0 and sw.s.mask.array[40, 50] > 0
        w.mask_panel.undo()
        assert sw.s.mask.pixel_count == before

    def test_polygon_outside_image_changes_nothing_and_adds_no_undo_step(self, sw):
        w = sw.w
        before = w.mask_panel.history.can_undo
        w._mask_lasso([(-80, -80), (-40, -80), (-40, -40), (-80, -40)], False)
        assert sw.s.mask.empty and w.mask_panel.history.can_undo == before

    def test_degenerate_polygon_reports_error_without_changing_mask(self, sw):
        w = sw.w
        w._mask_lasso([(10, 10), (20, 20)], False)
        assert sw.s.mask.empty and any(k == "error.unexpected" for k, _ in sw.errors)

    def test_tool_switch_restores_brush_and_hints_differ(self, sw):
        w, p = sw.w, sw.w.select_panel
        hints = set()
        for tool in ("mask_fill", "mask_erase", "lasso", "box", "brush"):
            p.set_tool(tool)
            hints.add(p.lbl_hint.text())
        assert w.canvas.select_mode is None and len(hints) == 5

    def test_shortcut_actions(self, sw):
        w = sw.w
        w.act_tool_mask_fill.trigger()
        assert w.select_panel.tool == "mask_fill" and w.canvas.select_mode == "mask_fill"
        w.act_tool_mask_erase.trigger()
        assert w.select_panel.tool == "mask_erase" and w.canvas.select_mode == "mask_erase"
        keys = {a.shortcut().toString() for a in (w.act_tool_mask_fill, w.act_tool_mask_erase)}
        assert keys == {"M", "D"}
        hebrew = {s.toString() for a in (w.act_tool_mask_fill, w.act_tool_mask_erase) for s in a.shortcuts()}
        assert {"צ", "ג"} <= hebrew

    def test_ignored_in_compare_mode(self, sw):
        w = sw.w
        w.select_panel.set_tool("mask_fill")
        w.act_compare.setChecked(True)
        assert not w.canvas.painting and not w.act_tool_mask_fill.isEnabled() and not w.act_tool_mask_erase.isEnabled()
        drag(w.canvas, SQUARE)
        w._mask_lasso(SQUARE, False)
        assert sw.s.mask.empty
        w.act_compare.setChecked(False)

    def test_no_session_is_harmless(self, app):
        from test_m34 import EWin
        h = EWin(engine=False)
        try:
            h.w._mask_lasso(SQUARE, False)
        finally:
            h.close()


class TestWithEngine:
    def test_fill_submits_job_and_commits_like_a_brush_stroke(self, ew):
        s = ew.install(h=200, w=300)
        before = s.rgb.copy()
        ew.w.select_panel.set_tool("mask_fill")
        drag(ew.canvas, SQUARE)
        assert wait_until(lambda: not np.array_equal(s.rgb, before) and not s.busy)
        assert s.mask.empty
        assert np.array_equal(s.rgb[0, 0], before[0, 0])         # מחוץ לאזור: ללא שינוי

    def test_erase_never_submits(self, ew):
        s = ew.install(h=200, w=300)
        s.mask.fill_polygon(SQUARE)
        before = s.rgb.copy()
        ew.w.select_panel.set_tool("mask_erase")
        drag(ew.canvas, [(40, 30), (200, 30), (200, 160), (40, 160)])
        wait_until(lambda: s.busy, 0.3)
        assert s.mask.empty and not s.busy and np.array_equal(s.rgb, before)
