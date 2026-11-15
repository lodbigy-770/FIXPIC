"""M4 / AUTO-SELECT שלב 4ב: חיווט GrabCut ל-Qt (מלבן, לאסו, משיכה מתקנת, הצעה, אישור). Qt offscreen.

מוכיח: מחוות הקנבס ואותותיו, ציור ההצעה בלי לגעת במסכה, בקר עם Provider מדומה (החלפת בקשות, כשל וחזרה אחורה, ריק, מחוון,
המתנה ל-Inpaint, זריקת הצעה אחרי שינוי תמונה), אישור כצעד Undo אחד בלי Inpaint, Esc לפי סדר, וקצה-לקצה עם GrabCut אמיתי
על תמונה סינתטית. **לא נבדק:** Windows, עכבר/טאבלט אמיתיים, מקלדת עברית (S/L/P/Enter לפי האות), מראה בעין, i5-6500T."""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.select import Context, Proposal, SelectionError  # noqa: E402
from fixpic.session import MaskHistory, Session  # noqa: E402
from fixpic.ui.canvas import SELECT_COLOR  # noqa: E402
from fixpic.ui.select_tool import SmartSelectController  # noqa: E402
from test_m34 import (EWin, T, app, drag, grab_rgb, painted_canvas, scr, wait_until, ew,  # noqa: E402,F401
                      _close_canvases)  # noqa: F401
from test_m4_grabcut import iou, scene  # noqa: E402

pytestmark = pytest.mark.usefixtures("_close_canvases")


# ------------------------------------------------------------------------------------------- עזרים
class FakeProvider:
    """Provider מדומה: מחזיר הצעה מלאה בגודל המלבן/המצולע שהתקבל, ורושם את כל הקלטים."""
    name = "fake"
    stale_ok = False

    def __init__(self, *, heavy=False, delay=0.0, fail_on=(), empty_on=(), gate: threading.Event | None = None):
        self.heavy, self.delay, self.fail_on, self.empty_on, self.gate = heavy, delay, set(fail_on), set(empty_on), gate
        self.calls: list = []
        self.started = threading.Event()

    def prepare(self, rgb, cancel=None):
        return Context(self.name, rgb.shape[:2], None, 0)

    def propose(self, ctx, prompts, cancel=None):
        n = len(self.calls)
        self.calls.append(prompts)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(T)
        if self.delay:
            time.sleep(self.delay)
        if cancel:
            cancel.raise_if_cancelled()
        if n in self.fail_on:
            raise SelectionError("נכשל בכוונה")
        if n in self.empty_on:
            return None
        if prompts.box is not None:
            x0, y0, x1, y1 = prompts.box
        else:
            xs, ys = [p[0] for p in prompts.polygon], [p[1] for p in prompts.polygon]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        return Proposal(np.full((y1 - y0, x1 - x0), 255, np.uint8), (x0, y0, x1, y1))


class Rig:
    """בקר עצמאי (בלי חלון) עם Provider מדומה, Session ו-MaskHistory; אותותיו נרשמים."""

    def __init__(self, provider, busy=None, h=120, w=200):
        self.session = Session(image_from_array(np.full((h, w, 3), 100, np.uint8)))
        self.history = MaskHistory(self.session.mask)
        self.ctl = SmartSelectController(provider_factory=lambda: provider, history_source=lambda: self.history,
                                         busy=busy)
        self.messages: list[str] = []
        self.indicator: list[bool] = []
        self.accepted: list = []
        self.ctl.message.connect(self.messages.append)
        self.ctl.processingChanged.connect(self.indicator.append)
        self.ctl.accepted.connect(self.accepted.append)
        self.ctl.bind(self.session)

    def settled(self, timeout=T):
        return wait_until(lambda: not self.ctl.processing, timeout)

    def close(self):
        self.ctl.shutdown()


@pytest.fixture()
def rig(app):
    made: list[Rig] = []

    def make(provider=None, **kw):
        r = Rig(provider or FakeProvider(), **kw)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


class SWin(EWin):
    """MainWindow עם תמונת `scene()` (320x240) וללא מנוע הסרה."""

    def __init__(self, provider_factory=None):
        super().__init__(engine=False)
        if provider_factory is not None:
            self.w.select_ctl._provider_factory = provider_factory
        img, self.ell, self.sq = scene()
        self.w._install(image_from_array(img), "scene.png")
        QApplication.processEvents()

    def proposal_done(self, timeout=T):
        return wait_until(lambda: self.w.select_ctl.proposal is not None and not self.w.select_ctl.processing, timeout)

    def close(self):
        self.w.select_ctl.shutdown()
        super().close()


@pytest.fixture()
def sw(app):
    h = SWin()
    yield h
    h.close()


def proposal_pixels(canvas) -> int:
    """כמה פיקסלים בקנבס משתנים בגלל ציור ההצעה (צילום עם הצעה מול צילום בלעדיה)."""
    prop = canvas._proposal
    canvas._hover = None
    with_p = grab_rgb(canvas).astype(int)
    canvas.set_proposal(None)
    without = grab_rgb(canvas).astype(int)
    canvas.set_proposal(prop)
    return int((abs(with_p - without).sum(axis=2) > 30).sum())


def green_count(canvas) -> int:
    px = grab_rgb(canvas).astype(int)
    r, g, b = SELECT_COLOR
    near = (abs(px[..., 0] - r) < 90) & (abs(px[..., 1] - g) < 60) & (abs(px[..., 2] - b) < 90) & (px[..., 1] > px[..., 0] + 40)
    return int(near.sum())


# --------------------------------------------------------------------------------- מחוות הקנבס
class TestCanvasGestures:
    def test_box_gesture_emits_normalized_image_rect_and_never_paints(self, app):
        c, layer = painted_canvas()
        c.set_select_mode("box")
        got, strokes = [], []
        c.selectBoxDrawn.connect(got.append)
        c.strokeStarted.connect(lambda *a: strokes.append(a))
        drag(c, [(150, 80), (40, 20)])                         # גרירה "הפוכה": x0<x1 ו-y0<y1 בכל זאת
        assert len(got) == 1 and strokes == [] and layer.empty
        x0, y0, x1, y1 = got[0]
        assert x0 < x1 and y0 < y1
        assert (x0, y0, x1, y1) == pytest.approx((40.5, 20.5, 150.5, 80.5), abs=1.0)

    def test_click_without_drag_is_not_a_box(self, app):
        c, _ = painted_canvas()
        c.set_select_mode("box")
        got = []
        c.selectBoxDrawn.connect(got.append)
        drag(c, [(50, 50), (50, 50)])
        drag(c, [(50, 50), (51, 50)])                          # פחות מ-4 פיקסלי מסך
        assert got == []

    def test_lasso_gesture_emits_polygon(self, app):
        c, layer = painted_canvas(h=200, w=300)
        c.set_select_mode("lasso")
        got = []
        c.selectLassoDrawn.connect(got.append)
        drag(c, [(50, 50), (150, 50), (150, 120), (50, 120)])
        assert len(got) == 1 and len(got[0]) >= 4 and layer.empty
        xs = [p[0] for p in got[0]]
        assert min(xs) == pytest.approx(50.5, abs=1.0) and max(xs) == pytest.approx(150.5, abs=1.0)

    def test_lasso_with_two_points_is_dropped(self, app):
        c, _ = painted_canvas()
        c.set_select_mode("lasso")
        got = []
        c.selectLassoDrawn.connect(got.append)
        drag(c, [(50, 50), (120, 50)])
        assert got == []

    def test_scribble_left_right_and_alt(self, app):
        c, layer = painted_canvas(h=200, w=300)
        c.set_select_mode("box")
        c.set_select_scribble(True)
        got = []
        c.selectScribbleDrawn.connect(lambda path, pos: got.append((path, pos)))
        drag(c, [(50, 50), (90, 50)])
        drag(c, [(50, 80), (90, 80)], button=Qt.RightButton)
        QTest.mousePress(c, Qt.LeftButton, Qt.AltModifier, scr(c, 50, 110))
        QTest.mouseMove(c, scr(c, 90, 110))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.AltModifier, scr(c, 90, 110))
        assert [pos for _, pos in got] == [True, False, False]
        assert all(len(path) >= 2 for path, _ in got) and layer.empty

    def test_shift_drag_starts_a_new_box_even_with_active_input(self, app):
        c, _ = painted_canvas(h=200, w=300)
        c.set_select_mode("box")
        c.set_select_scribble(True)
        boxes, scribbles = [], []
        c.selectBoxDrawn.connect(boxes.append)
        c.selectScribbleDrawn.connect(lambda *a: scribbles.append(a))
        QTest.mousePress(c, Qt.LeftButton, Qt.ShiftModifier, scr(c, 30, 30))
        QTest.mouseMove(c, scr(c, 120, 90))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.ShiftModifier, scr(c, 120, 90))
        assert len(boxes) == 1 and scribbles == []

    def test_right_button_without_input_does_nothing(self, app):
        c, layer = painted_canvas()
        c.set_select_mode("box")
        got = []
        c.selectBoxDrawn.connect(got.append)
        drag(c, [(30, 30), (90, 70)], button=Qt.RightButton)
        assert got == [] and layer.empty and not c.gesture_active

    def test_cancel_gesture_mid_drag_emits_nothing(self, app):
        c, _ = painted_canvas()
        c.set_select_mode("box")
        got = []
        c.selectBoxDrawn.connect(got.append)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, scr(c, 30, 30))
        QTest.mouseMove(c, scr(c, 100, 70))
        assert c.gesture_active
        assert c.cancel_gesture() is True and c.cancel_gesture() is False
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, scr(c, 100, 70))
        assert got == []

    def test_gestures_only_in_painting_mode(self, app):
        c, _ = painted_canvas()
        c.set_select_mode("box")
        c.set_painting(False)                                   # עסוק/השוואה: כפתור שמאל מזיז
        got = []
        c.selectBoxDrawn.connect(got.append)
        before = QPointF(c.offset)
        drag(c, [(60, 40), (100, 60)])
        assert got == [] and c.offset != before

    def test_turning_painting_off_mid_gesture_cancels_it(self, app):
        c, _ = painted_canvas()
        c.set_select_mode("lasso")
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, scr(c, 30, 30))
        QTest.mouseMove(c, scr(c, 60, 60))
        c.set_painting(False)
        assert not c.gesture_active

    def test_mode_none_restores_brush_strokes(self, app):
        c, layer = painted_canvas()
        c.set_select_mode("box")
        c.set_select_mode(None)
        started = []
        c.strokeStarted.connect(lambda *a: started.append(a))
        drag(c, [(50, 50), (80, 50)])
        assert len(started) == 1

    def test_invalid_mode(self, app):
        c, _ = painted_canvas()
        with pytest.raises(ValueError):
            c.set_select_mode("wand")


class TestProposalOverlay:
    def test_proposal_drawn_green_at_its_rect_without_touching_the_mask(self, app):
        c, layer = painted_canvas(h=100, w=200)
        c.scale, c.offset, c._fit_mode = 1.0, QPointF(0, 0), False
        win = np.full((30, 40), 255, np.uint8)
        c.set_proposal(Proposal(win, (50, 20, 90, 50)))
        QApplication.processEvents()
        c._hover = None
        shot = grab_rgb(c).astype(int)
        inside, outside = shot[35, 70], shot[80, 150]
        assert inside[1] > inside[0] + 40 and abs(outside[0] - 100) < 5      # ירוק בפנים, רקע בחוץ
        assert shot[35, 49][1] < shot[35, 49][0] + 40 and layer.empty
        c.set_proposal(None)
        c._hover = None
        assert abs(grab_rgb(c)[35, 70][0] - 100) < 5

    def test_proposal_partly_outside_view_and_zoomed(self, app):
        c, _ = painted_canvas(h=100, w=200)
        c.scale, c.offset, c._fit_mode = 4.0, QPointF(-100, -20), False
        c.set_proposal(Proposal(np.full((30, 40), 255, np.uint8), (50, 20, 90, 50)))
        QApplication.processEvents()
        c._hover = None
        assert green_count(c) > 1000

    def test_same_proposal_does_not_invalidate_cache(self, app):
        c, _ = painted_canvas()
        p = Proposal(np.full((10, 10), 255, np.uint8), (5, 5, 15, 15))
        c.set_proposal(p)
        QApplication.processEvents()
        items = c._ov_cache
        c.set_proposal(p)
        assert c._ov_cache is items


# ------------------------------------------------------------------------------------------ הבקר
class TestController:
    def test_box_request_produces_proposal_and_does_not_touch_mask(self, rig):
        r = rig()
        assert r.ctl.request_box((10.2, 20.7, 90.1, 80.3))
        assert r.settled() and r.ctl.proposal is not None
        assert r.ctl.proposal.rect == (10, 20, 91, 81)                      # עיגול החוצה
        assert r.session.mask.empty and not r.history.can_undo
        assert r.ctl.has_input and r.ctl.has_proposal

    def test_tiny_box_and_lasso_rejected_with_message(self, rig):
        r = rig()
        assert not r.ctl.request_box((10, 10, 14, 40))
        assert not r.ctl.request_lasso([(10, 10), (12, 11), (11, 40)])
        assert not r.ctl.request_lasso([(10, 10), (50, 50)])
        assert r.messages.count("האזור קטן מדי לבחירה אוטומטית.") == 3 and not r.ctl.has_input

    def test_box_clipped_to_image_and_garbage_ignored(self, rig):
        r = rig()
        assert not r.ctl.request_box((float("nan"), 0, 50, 50))
        assert not r.ctl.request_box(("a", 0, 50, 50))
        assert r.ctl.request_box((-50, -50, 5000, 5000))
        assert r.settled() and r.ctl.proposal.rect == (0, 0, 200, 120)

    def test_lasso_request_clamps_points_to_image(self, rig):
        prov = FakeProvider()
        r = rig(prov)
        assert r.ctl.request_lasso([(-30, -30), (100, -10), (100, 80), (-20, 70)])
        assert r.settled()
        assert min(p[0] for p in prov.calls[0].polygon) == 0 and min(p[1] for p in prov.calls[0].polygon) == 0

    def test_scribbles_accumulate_on_the_same_input(self, rig):
        prov = FakeProvider()
        r = rig(prov)
        r.ctl.request_box((10, 10, 150, 100))
        assert r.settled()
        assert r.ctl.request_scribble([(30, 30), (60, 30)], True, 4)
        assert r.settled()
        n1 = len(prov.calls[1].points)
        assert n1 > 0 and prov.calls[1].box == (10, 10, 150, 100) and all(p.positive for p in prov.calls[1].points)
        assert r.ctl.request_scribble([(100, 60), (120, 60)], False, 4)
        assert r.settled()
        pts = prov.calls[2].points
        assert len(pts) > n1 and any(not p.positive for p in pts) and prov.calls[2].box == (10, 10, 150, 100)

    def test_scribble_requires_input_and_inside_image(self, rig):
        r = rig()
        assert not r.ctl.request_scribble([(30, 30), (60, 30)], True, 4)     # אין קלט פעיל
        r.ctl.request_box((10, 10, 150, 100))
        assert r.settled()
        assert not r.ctl.request_scribble([(-90, -90), (-60, -90)], True, 4)
        assert "המשיכה מחוץ לתמונה." in r.messages

    def test_new_request_replaces_running_one(self, rig):
        gate = threading.Event()
        prov = FakeProvider(gate=gate)
        r = rig(prov)
        r.ctl.request_box((10, 10, 60, 60))
        assert prov.started.wait(T)
        r.ctl.request_box((20, 20, 120, 100))
        gate.set()
        assert r.settled()
        assert wait_until(lambda: r.ctl.proposal is not None)
        assert r.ctl.proposal.rect == (20, 20, 120, 100)

    def test_failure_rolls_back_to_previous_proposal_and_input(self, rig):
        prov = FakeProvider(fail_on={1})
        r = rig(prov)
        r.ctl.request_box((10, 10, 150, 100))
        assert r.settled()
        first = r.ctl.proposal
        r.ctl.request_scribble([(30, 30), (60, 30)], True, 4)
        assert r.settled()
        assert r.ctl.proposal is first and r.ctl.has_input
        assert any(m.startswith("הבחירה נכשלה") for m in r.messages)
        n_failed = len(prov.calls[1].points)
        r.ctl.request_scribble([(30, 50), (60, 50)], True, 4)                  # הקלט המצטבר לא כולל את המשיכה שנכשלה
        assert r.settled()
        assert len(prov.calls[2].points) == n_failed                            # אותה משיכה בגודלה, ולא שתיים

    def test_failure_of_first_request_leaves_nothing_active(self, rig):
        r = rig(FakeProvider(fail_on={0}))
        r.ctl.request_box((10, 10, 150, 100))
        assert r.settled()
        assert r.ctl.proposal is None and not r.ctl.has_input and not r.ctl.active

    def test_empty_result_clears_proposal_but_keeps_input_for_correction(self, rig):
        prov = FakeProvider(empty_on={1})
        r = rig(prov)
        r.ctl.request_box((10, 10, 150, 100))
        assert r.settled()
        r.ctl.request_scribble([(30, 30), (60, 30)], False, 4)
        assert r.settled()
        assert r.ctl.proposal is None and r.ctl.has_input
        assert "לא נמצא אובייקט בבחירה. אפשר לתקן במשיכה או לצייר מלבן אחר." in r.messages
        assert r.ctl.request_scribble([(100, 60), (120, 60)], True, 4)         # אפשר להמשיך לתקן
        assert r.settled() and r.ctl.proposal is not None

    def test_accept_is_one_mask_undo_step_and_clears_state(self, rig):
        r = rig()
        r.ctl.request_box((10, 20, 60, 70))
        assert r.settled()
        rect = r.ctl.accept()
        assert rect == (10, 20, 60, 70) and r.accepted == [rect]
        m = r.session.mask.array
        assert m[20:70, 10:60].all() and int(m.sum() // 255) == 50 * 50
        assert r.history.undo_count == 1 and not r.ctl.active
        r.history.undo()
        assert r.session.mask.empty
        assert r.session.history.current_id == 0 and not r.session.busy         # בלי Commit ובלי Inpaint

    def test_accept_without_proposal_and_when_already_in_mask(self, rig):
        r = rig()
        assert r.ctl.accept() is None
        r.session.mask.add_window((10, 20, 60, 70), np.full((50, 50), 255, np.uint8))
        r.ctl.request_box((10, 20, 60, 70))
        assert r.settled()
        assert r.ctl.accept() is None
        assert "הבחירה כבר כלולה במסכה." in r.messages and not r.ctl.active

    def test_cancel_drops_everything_and_cancels_running(self, rig):
        gate = threading.Event()
        prov = FakeProvider(gate=gate)
        r = rig(prov)
        r.ctl.request_box((10, 10, 60, 60))
        assert prov.started.wait(T)
        assert r.ctl.cancel() is True and not r.ctl.active
        gate.set()
        time.sleep(0.1)
        QApplication.processEvents()
        assert r.ctl.proposal is None and not r.ctl.has_input                    # התוצאה שבוטלה לא חוזרת
        assert r.ctl.cancel() is False

    def test_image_change_drops_proposal(self, rig):
        r = rig()
        r.ctl.request_box((10, 10, 60, 60))
        assert r.settled()
        r.ctl.on_image_changed()
        assert not r.ctl.active

    def test_bind_new_session_drops_state(self, rig):
        r = rig()
        r.ctl.request_box((10, 10, 60, 60))
        assert r.settled()
        r.ctl.bind(Session(image_from_array(np.full((50, 50, 3), 1, np.uint8))))
        assert not r.ctl.active

    def test_indicator_only_after_200ms_and_turns_off(self, rig):
        r = rig(FakeProvider(delay=0.6))
        r.ctl.request_box((10, 10, 60, 60))
        time.sleep(0.05)
        QApplication.processEvents()
        assert r.indicator == []                                                 # מהיר מ-200ms: אין מחוון
        assert wait_until(lambda: r.indicator == [True])
        assert r.settled() and r.indicator == [True, False]

    def test_fast_selection_never_shows_indicator(self, rig):
        r = rig()
        r.ctl.request_box((10, 10, 60, 60))
        assert r.settled()
        QTest.qWait(300)
        assert r.indicator == []

    def test_heavy_provider_waits_for_inpaint_then_runs(self, rig):
        busy = {"on": True}
        prov = FakeProvider(heavy=True)
        r = rig(prov, busy=lambda: busy["on"])
        r.ctl.request_box((10, 10, 60, 60))
        time.sleep(0.3)
        assert prov.calls == [] and r.ctl.processing
        busy["on"] = False
        assert r.settled() and len(prov.calls) == 1 and r.ctl.proposal is not None

    def test_context_key_changes_with_history(self, rig):
        """Context של GrabCut תלוי בצבעים: מפתח לפי (Session, history.current_id), ולכן אחרי Commit נבנה חדש."""
        r = rig()
        r.ctl.request_box((10, 10, 60, 60))
        assert r.settled()
        keys = r.ctl._service.context_keys
        assert keys == [(id(r.session), r.session.history.current_id)]

    def test_shutdown_is_idempotent_and_service_restarts_lazily(self, rig):
        r = rig()
        r.ctl.request_box((10, 10, 60, 60))
        assert r.settled()
        r.ctl.shutdown()
        r.ctl.shutdown()
        assert r.ctl._service is None
        assert r.ctl.request_box((10, 10, 60, 60)) and r.settled() and r.ctl.proposal is not None

    def test_no_session_is_harmless(self, app):
        ctl = SmartSelectController(provider_factory=FakeProvider)
        assert not ctl.request_box((0, 0, 50, 50)) and not ctl.request_lasso([(0, 0), (50, 0), (50, 50)])
        assert not ctl.request_scribble([(1, 1)], True, 3) and ctl.accept() is None and ctl.cancel() is False
        ctl.on_image_changed()
        ctl.shutdown()


# -------------------------------------------------------------------------- חלון ראשי + GrabCut אמיתי
class TestWindowEndToEnd:
    def test_box_over_both_objects_proposes_then_enter_accepts_one_undo_step(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        assert cv.select_mode == "box"
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        prop = w.select_ctl.proposal
        assert iou(_full(prop, sw.s.size[::-1]), sw.ell | sw.sq) > 0.85
        assert sw.s.mask.empty and proposal_pixels(cv) > 3000                    # מצויר מעל התמונה, לא במסכה
        assert w.act_select_accept.isEnabled() and w.select_panel.btn_accept.isEnabled()
        w.act_select_accept.trigger()
        assert not sw.s.mask.empty and not w.select_ctl.active and not w.act_select_accept.isEnabled()
        assert w.mask_panel.history.undo_count == 1 and not sw.s.busy and sw.s.history.current_id == 0
        w.mask_panel.undo()
        assert sw.s.mask.empty

    def test_negative_scribble_removes_the_unwanted_object(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        assert cv.select_scribble
        w.brush_panel.set_diameter(20)
        drag(cv, [(215, 150), (265, 150), (265, 195), (215, 195)], button=Qt.RightButton)     # על הריבוע הכחול
        assert wait_until(lambda: not w.select_ctl.processing and w.select_ctl.proposal is not None)
        m = _full(w.select_ctl.proposal, sw.s.size[::-1])
        assert iou(m, sw.ell) > 0.8 and (m & sw.sq).sum() < 0.2 * sw.sq.sum()

    def test_positive_scribble_after_tight_box_adds_missing_part(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(60, 60), (160, 140)])                       # מלבן שחותך את האליפסה
        assert sw.proposal_done()
        assert w.select_ctl.has_input

    def test_lasso_end_to_end(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("lasso")
        assert cv.select_mode == "lasso"
        drag(cv, [(55, 55), (110, 50), (165, 70), (170, 100), (160, 135), (110, 150), (60, 135), (50, 100)])
        assert sw.proposal_done()
        assert iou(_full(w.select_ctl.proposal, sw.s.size[::-1]), sw.ell) > 0.8

    def test_escape_order_gesture_then_selection_then_inpaint(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        QTest.mousePress(cv, Qt.LeftButton, Qt.NoModifier, scr(cv, 40, 40))
        QTest.mouseMove(cv, scr(cv, 200, 150))
        assert cv.gesture_active
        w._escape()
        assert not cv.gesture_active and not w.select_ctl.active                   # מחוות בוטלה, עדיין אין קלט
        QTest.mouseRelease(cv, Qt.LeftButton, Qt.NoModifier, scr(cv, 200, 150))
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        w._escape()
        assert not w.select_ctl.active and cv._proposal is None and not cv.select_scribble
        assert w.select_panel.tool == "box"                                          # הכלי נשאר

    def test_escape_cancels_inpaint_when_no_selection(self, ew):
        ew.install()
        called = []
        ew.w.controller.cancel = lambda: called.append(1) or False
        ew.w._escape()
        assert called == [1]

    def test_switching_to_brush_drops_selection_and_restores_painting(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        w.select_panel.set_tool("brush")
        assert not w.select_ctl.active and cv.select_mode is None and cv._proposal is None
        drag(cv, [(60, 60), (90, 60)])
        assert not sw.s.mask.empty                                                   # מברשת מצייר כרגיל

    def test_select_drag_does_not_paint_or_submit_jobs(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        assert sw.s.mask.empty and not sw.s.busy and w.controller.waiting is None

    def test_tool_shortcut_actions(self, sw):
        w = sw.w
        w.act_tool_box.trigger()
        assert w.select_panel.tool == "box" and w.canvas.select_mode == "box"
        w.act_tool_lasso.trigger()
        assert w.select_panel.tool == "lasso" and w.canvas.select_mode == "lasso"
        w.act_tool_brush.trigger()
        assert w.select_panel.tool == "brush" and w.canvas.select_mode is None
        assert {a.shortcut().toString() for a in (w.act_tool_box, w.act_tool_lasso, w.act_tool_brush)} == {"S", "L", "P"}

    def test_image_commit_or_undo_signal_drops_proposal(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        w.controller.imageChanged.emit()
        assert not w.select_ctl.active and cv._proposal is None

    def test_new_image_resets_selection(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        drag(cv, [(40, 40), (290, 215)])
        assert sw.proposal_done()
        img, _, _ = scene(seed=5)
        w._install(image_from_array(img), "other.png")
        QApplication.processEvents()
        assert not w.select_ctl.active and cv._proposal is None and w.select_panel.tool == "box"

    def test_compare_mode_disables_selection_input(self, sw):
        w, cv = sw.w, sw.canvas
        w.select_panel.set_tool("box")
        w.act_compare.setChecked(True)
        assert not cv.painting and not w.select_panel.isEnabled() and not w.act_tool_box.isEnabled()
        drag(cv, [(40, 40), (290, 215)])
        assert not w.select_ctl.active
        w.act_compare.setChecked(False)
        assert cv.painting and w.select_panel.isEnabled()

    def test_without_session_everything_disabled(self, app):
        h = EWin(engine=False)
        try:
            w = h.w
            assert not w.select_panel.isEnabled() and not w.act_select_accept.isEnabled()
            assert not w.act_tool_box.isEnabled()
            w._select_accept()
            w._escape()
        finally:
            h.close()

    def test_indicator_label_follows_controller(self, sw):
        w = sw.w
        w.select_ctl.processingChanged.emit(True)
        assert w.lbl_sel.isVisibleTo(w.statusBar())
        w.select_ctl.processingChanged.emit(False)
        assert not w.lbl_sel.isVisibleTo(w.statusBar())

    def test_panel_state_and_hints(self, sw):
        p = sw.w.select_panel
        assert not p.btn_accept.isEnabled() and not p.btn_cancel.isEnabled()
        h_brush = p.lbl_hint.text()
        p.set_tool("box")
        h_box = p.lbl_hint.text()
        p.set_tool("lasso")
        assert len({h_brush, h_box, p.lbl_hint.text()}) == 3
        p.set_state(False, True)
        assert p.btn_cancel.isEnabled() and not p.btn_accept.isEnabled()
        p.set_state(True, True)
        assert p.btn_accept.isEnabled()
        with pytest.raises(ValueError):
            p.set_tool("wand")

    def test_close_shuts_down_service(self, app):
        h = SWin()
        w = h.w
        w.select_panel.set_tool("box")
        drag(h.canvas, [(40, 40), (290, 215)])
        assert h.proposal_done()
        ctl = w.select_ctl
        ctl.shutdown()
        assert ctl._service is None
        h.close()


def _full(prop, shape):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = prop.rect
    m[y0:y1, x0:x1] = prop.window > 0
    return m
