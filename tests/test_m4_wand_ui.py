"""M4 / AUTO-SELECT שלב 4א: כלי השרביט בממשק, מעל מבנה 4ב (`SmartSelectController`). Qt offscreen, `MagicWandProvider` אמיתי,
ו-Provider מדומה לתרחישי שגיאה/עיכוב.

מוכיח: לחיצה → הצעה בירוק בלי לגעת במסכה/היסטוריה/Session; Enter מאשר (מוסיף למסכה כצעד אחד ומגיש Inpaint, כפי שנקבע ב-4ב); גרירה אופקית וסליידר משנים
סבילות בתצוגה חיה (ספירת פיקסלים מדויקת על תמונת פסים); Esc/החלפת כלי/שינוי תמונה מבטלים הצעה; והשרביט ו-GrabCut אינם מפריעים זה
לזה. **לא נבדק:** Windows, מקלדת עברית (`W`/`P`/`Enter` לפי האות), תחושת הגרירה בעכבר אמיתי, תמונות אמיתיות (המדידות ב-
`tools/m4/wand_samples.py`), 24MP בממשק, i5-6500T."""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.select import Context, Proposal, SelectionError  # noqa: E402
from fixpic.session import Session  # noqa: E402
from fixpic.strings import t  # noqa: E402
from fixpic.ui.canvas import Canvas  # noqa: E402
from fixpic.ui.select_panel import SelectPanel  # noqa: E402
from fixpic.ui.wand_tool import DRAG_PX_PER_STEP, MAX_UI_TOLERANCE, WandController  # noqa: E402
from test_engine import FILL, rnd  # noqa: E402
from test_m34 import (_CANVASES, EWin, T, app, ew, ew_noengine, grab_rgb, painted_canvas, scr,  # noqa: E402,F401
                      wait_until, _close_canvases)  # noqa: F401
from test_m35 import view_1to1  # noqa: E402

pytestmark = pytest.mark.usefixtures("_close_canvases")


# ------------------------------------------------------------------------------------------------ תמונות וכלים
def two_halves(h=200, w=300, a=100, b=200) -> np.ndarray:
    """חצי שמאלי אפור כהה אחיד וימני בהיר אחיד (בלי רעש: floodFill מדויק)."""
    arr = np.empty((h, w, 3), np.uint8)
    arr[:, : w // 2] = a
    arr[:, w // 2:] = b
    return arr


def bands(h=200, w=300) -> np.ndarray:
    """10 פסים אנכיים ברוחב 30: ערך 100 + 6*i (הפרש מדויק מול הפס הראשון)."""
    arr = np.empty((h, w, 3), np.uint8)
    for i in range(10):
        arr[:, i * 30:(i + 1) * 30] = 100 + 6 * i
    return arr


def install(ew, arr):
    ew.w._install(image_from_array(arr), "x.png")
    QApplication.processEvents()
    view_1to1(ew.canvas)
    ew.canvas._hover = None
    return ew.s


def wand(ew):
    ew.w.act_tool_wand.trigger()
    QApplication.processEvents()


def click(canvas, x, y):
    """לחיצה על פיקסל מסך (בזום 100% והזזה 0: זהה לפיקסל תמונה)."""
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(x, y))
    QApplication.processEvents()


def ctl_of(ew) -> WandController:
    return ew.w.wand_ctl


def proposal_is(ew, count=None, rect=None, timeout=T):
    def ok():
        p = ctl_of(ew).proposal
        return p is not None and (count is None or p.pixel_count == count) and (rect is None or p.rect == rect)
    return wait_until(ok, timeout)


def key(ew, k, mod=Qt.NoModifier):
    ew.w.activateWindow()
    ew.canvas.setFocus()
    QApplication.processEvents()
    QTest.keyClick(ew.canvas, k, mod)
    QApplication.processEvents()


def status(ew) -> str:
    return ew.w.statusBar().currentMessage()


# ------------------------------------------------------------------------------------------------ Canvas
class TestCanvasWandMode:
    def make(self, app, w=300, h=200):
        c = Canvas()
        c.resize(400, 300)
        c.show()
        c.set_image(two_halves(h, w))
        c.set_painting(True)
        c.set_select_mode("wand")
        view_1to1(c)
        _CANVASES.append(c)
        return c

    def test_press_emits_continuous_image_point_inside_only(self, app):
        c = self.make(app)
        got, other = [], []
        c.wandPressed.connect(got.append)
        c.strokeStarted.connect(lambda *a: other.append("stroke"))
        c.selectBoxDrawn.connect(lambda *a: other.append("box"))
        c.selectScribbleDrawn.connect(lambda *a: other.append("scribble"))
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, QPoint(120, 80))
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, QPoint(350, 80))      # מחוץ לתמונה (רוחב 300)
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, QPoint(120, 250))     # מחוץ לתמונה (גובה 200)
        assert len(got) == 1 and (got[0].x(), got[0].y()) == (120.0, 80.0) and other == []

    def test_right_button_does_nothing(self, app):
        c = self.make(app)
        got, other = [], []
        c.wandPressed.connect(got.append)
        c.strokeStarted.connect(lambda *a: other.append(1))
        c.selectScribbleDrawn.connect(lambda *a: other.append(2))
        QTest.mouseClick(c, Qt.RightButton, Qt.NoModifier, QPoint(50, 50))
        assert got == [] and other == [] and not c.gesture_active

    def test_scribble_flag_does_not_turn_a_wand_drag_into_a_scribble(self, app):
        c = self.make(app)
        c.set_select_scribble(True)                       # גם אם מישהו הדליק (לשרביט החלון לא מדליק)
        ev = []
        c.wandPressed.connect(lambda p: ev.append("press"))
        c.selectScribbleDrawn.connect(lambda *a: ev.append("scribble"))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        QTest.mouseMove(c, QPoint(140, 100))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(140, 100))
        assert ev == ["press"]

    def test_drag_reports_horizontal_offset_then_release(self, app):
        c = self.make(app)
        ev = []
        c.wandPressed.connect(lambda p: ev.append("press"))
        c.wandDragged.connect(lambda dx: ev.append(round(dx)))
        c.wandReleased.connect(lambda: ev.append("release"))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        QTest.mouseMove(c, QPoint(140, 100))
        QTest.mouseMove(c, QPoint(90, 130))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(90, 130))
        assert ev[0] == "press" and 40 in ev and -10 in ev and ev[-1] == "release" and not c.gesture_active

    def test_mode_switch_or_painting_off_mid_drag_drops_the_gesture_silently(self, app):
        c = self.make(app)
        ev = []
        c.wandDragged.connect(lambda dx: ev.append(dx))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        assert c.gesture_active
        c.set_select_mode(None)
        assert not c.gesture_active
        QTest.mouseMove(c, QPoint(140, 100))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(140, 100))
        c.set_select_mode("wand")
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        c.set_painting(False)
        assert not c.gesture_active and ev == []

    def test_brush_mode_still_strokes(self, app):
        c = self.make(app)
        c.set_select_mode(None)
        starts, presses = [], []
        c.strokeStarted.connect(lambda *a: starts.append(a))
        c.wandPressed.connect(presses.append)
        QTest.mouseClick(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        assert len(starts) == 1 and presses == []

    def test_cursor_is_cross_in_wand_mode_and_no_brush_ring(self, app):
        c = self.make(app)
        assert c.cursor().shape() == Qt.CrossCursor
        c.set_select_mode(None)
        assert c.cursor().shape() == Qt.BlankCursor

    def test_seed_ring_is_drawn_only_in_wand_mode_and_cleared(self, app):
        c = self.make(app)
        c._hover = None
        base = grab_rgb(c)[:200, :300].astype(int)
        c.set_wand_seed(QPointF(150.0, 100.0))
        c._hover = None
        ring = grab_rgb(c)[:200, :300].astype(int)
        diff = (abs(ring - base).sum(axis=2) > 30)
        assert diff.sum() > 20 and diff[100, 144:157].any()               # טבעת ברדיוס ~6 סביב הלחיצה
        assert not diff[:, :130].any() and not diff[:, 170:].any()
        c.set_select_mode("box")
        c._hover = None
        assert (abs(grab_rgb(c)[:200, :300].astype(int) - base).sum(axis=2) > 30).sum() == 0
        c.set_select_mode("wand")
        c.set_wand_seed(None)
        c._hover = None
        assert (abs(grab_rgb(c)[:200, :300].astype(int) - base).sum(axis=2) > 30).sum() == 0

    def test_wand_gesture_itself_paints_nothing(self, app):
        c = self.make(app)
        c._hover = None
        base = grab_rgb(c)[:200, :300].astype(int)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        QTest.mouseMove(c, QPoint(160, 100))
        c._hover = None
        mid = grab_rgb(c)[:200, :300].astype(int)
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(160, 100))
        assert (abs(mid - base).sum(axis=2) > 30).sum() == 0                # בלי מלבן/קו תוך כדי גרירה


# ------------------------------------------------------------------------------------------------ לוח
class TestPanel:
    def test_wand_is_the_second_tool_with_its_own_hint(self, app):
        p = SelectPanel()
        assert list(p.radios)[:4] == ["brush", "wand", "box", "lasso"]
        hints = []
        for tool in ("brush", "wand", "box", "lasso"):
            p.set_tool(tool)
            hints.append(p.lbl_hint.text())
        assert len(set(hints)) == 4 and hints[1] == t("select.hint_wand")

    def test_tolerance_controls_enabled_only_for_the_wand(self, app):
        p = SelectPanel()
        assert not p.slider.isEnabled() and not p.spin.isEnabled()
        p.set_tool("wand")
        assert p.slider.isEnabled() and p.spin.isEnabled()
        p.set_tool("box")
        assert not p.slider.isEnabled()

    def test_user_changes_signal_but_set_tolerance_is_silent_and_synced(self, app):
        p = SelectPanel()
        got = []
        p.toleranceChanged.connect(got.append)
        p.set_tolerance(33)
        assert got == [] and p.slider.value() == 33 and p.spin.value() == 33 and p.tolerance == 33
        p.slider.setValue(40)
        assert got == [40] and p.spin.value() == 40
        p.spin.setValue(7)
        assert got == [40, 7] and p.slider.value() == 7

    def test_range_is_0_to_max(self, app):
        p = SelectPanel()
        assert (p.slider.minimum(), p.slider.maximum(), p.spin.maximum()) == (0, MAX_UI_TOLERANCE, MAX_UI_TOLERANCE)


# ------------------------------------------------------------------------------------------------ בקר (Provider מדומה)
class FakeProv:
    name = "fake"
    heavy = False
    stale_ok = False

    def __init__(self):
        self.calls: list = []
        self.gate: threading.Event | None = None
        self.delay = 0.0
        self.boom: BaseException | None = None
        self.result = None

    def prepare(self, rgb, cancel=None):
        return Context(self.name, rgb.shape[:2], None, 1)

    def propose(self, ctx, prompts, cancel=None):
        self.calls.append(prompts)
        if self.gate is not None:
            self.gate.wait(5)
        if self.delay:
            time.sleep(self.delay)
        if cancel:
            cancel.raise_if_cancelled()
        if self.boom:
            raise self.boom
        return self.result(prompts) if callable(self.result) else self.result


def box_prop(rect=(10, 10, 30, 30)):
    x0, y0, x1, y1 = rect
    return Proposal(np.full((y1 - y0, x1 - x0), 255, np.uint8), rect)


class Ctl:
    def __init__(self, prov):
        self.prov = prov
        self.c = WandController(provider_factory=lambda: prov)
        self.session = Session(image_from_array(rnd(100, 120, 1)))
        self.c.bind(self.session)
        self.msgs, self.busy, self.accepted, self.tols = [], [], [], []
        self.c.message.connect(self.msgs.append)
        self.c.processingChanged.connect(self.busy.append)
        self.c.accepted.connect(self.accepted.append)
        self.c.toleranceChanged.connect(self.tols.append)

    def close(self):
        if self.prov.gate is not None:
            self.prov.gate.set()
        self.c.shutdown()


@pytest.fixture()
def ctl(app):
    h = Ctl(FakeProv())
    yield h
    h.close()


class TestController:
    def test_press_requests_a_proposal_with_the_tolerance_and_exposes_the_seed(self, ctl):
        ctl.prov.result = box_prop()
        assert ctl.c.press(QPointF(33.5, 44.25)) is True
        assert wait_until(lambda: ctl.c.proposal is not None)
        pr = ctl.prov.calls[-1]
        assert [(p.x, p.y, p.positive) for p in pr.points] == [(33.5, 44.25, True)] and pr.tolerance == 20
        assert (ctl.c.seed.x(), ctl.c.seed.y()) == (33.5, 44.25) and ctl.c.has_input and ctl.c.has_proposal
        assert t("select.proposal_wand", count=400) in ctl.msgs

    def test_press_without_a_session_is_ignored(self, app):
        c = WandController(provider_factory=lambda: FakeProv())
        assert c.press(QPointF(1, 1)) is False and not c.active and c.seed is None
        c.shutdown()

    def test_empty_result_keeps_the_seed_and_says_so_in_wand_words(self, ctl):
        ctl.prov.result = None
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: t("select.empty_wand") in ctl.msgs)
        assert ctl.c.proposal is None and ctl.c.seed is not None

    def test_failure_rolls_back_to_the_previous_proposal_and_reports(self, ctl):
        ctl.prov.result = box_prop((0, 0, 10, 10))
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: ctl.c.proposal is not None)
        ctl.prov.boom = SelectionError("נקודה מחוץ לתמונה")
        ctl.c.press(QPointF(6, 6))
        assert wait_until(lambda: any("נקודה מחוץ לתמונה" in m for m in ctl.msgs))
        assert ctl.c.proposal is not None and ctl.c.proposal.rect == (0, 0, 10, 10)
        assert (ctl.c.seed.x(), ctl.c.seed.y()) == (5.0, 5.0)

    def test_new_input_replaces_a_slow_request_and_the_old_result_is_dropped(self, ctl):
        ctl.prov.gate = threading.Event()
        ctl.prov.result = lambda pr: box_prop((0, 0, 10, 10)) if pr.points[0].x == 1 else box_prop((50, 50, 60, 60))
        ctl.c.press(QPointF(1, 1))
        assert wait_until(lambda: len(ctl.prov.calls) == 1)
        ctl.c.press(QPointF(2, 2))
        ctl.prov.gate.set()
        assert wait_until(lambda: ctl.c.proposal is not None)
        time.sleep(0.1)
        QApplication.processEvents()
        assert ctl.c.proposal.rect == (50, 50, 60, 60)

    def test_drag_sets_tolerance_from_the_press_baseline_and_requests_live(self, ctl):
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: ctl.c.proposal is not None)
        ctl.c.drag(4 * DRAG_PX_PER_STEP)
        assert ctl.c.tolerance == 24 and ctl.tols == [24]
        assert wait_until(lambda: ctl.prov.calls[-1].tolerance == 24)
        ctl.c.drag(-2 * DRAG_PX_PER_STEP)
        assert ctl.c.tolerance == 18
        ctl.c.press(QPointF(5, 5))                           # לחיצה חדשה: הבסיס הוא 18
        ctl.c.drag(DRAG_PX_PER_STEP)
        assert ctl.c.tolerance == 19

    def test_drag_without_a_seed_is_ignored(self, ctl):
        ctl.c.drag(100)
        assert ctl.c.tolerance == 20 and ctl.tols == [] and ctl.prov.calls == []

    def test_tolerance_is_clamped(self, ctl):
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        ctl.c.drag(10_000)
        assert ctl.c.tolerance == MAX_UI_TOLERANCE
        ctl.c.drag(-10_000)
        assert ctl.c.tolerance == 0
        ctl.c.set_tolerance(-5)
        assert ctl.c.tolerance == 0

    def test_set_tolerance_without_a_seed_only_stores_it(self, ctl):
        ctl.c.set_tolerance(33)
        time.sleep(0.05)
        assert ctl.c.tolerance == 33 and ctl.prov.calls == [] and not ctl.c.active and ctl.tols == [33]

    def test_unchanged_tolerance_does_not_rerequest(self, ctl):
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: ctl.c.proposal is not None)
        n = len(ctl.prov.calls)
        ctl.c.set_tolerance(20)
        ctl.c.drag(1)                                        # round(1/4) = 0
        time.sleep(0.05)
        assert len(ctl.prov.calls) == n and ctl.tols == []

    def test_accept_adds_the_proposal_to_the_session_mask_once(self, ctl):
        ctl.prov.result = box_prop((10, 10, 30, 30))
        ctl.c.press(QPointF(15, 15))
        assert wait_until(lambda: ctl.c.proposal is not None)
        assert ctl.c.accept() == (10, 10, 30, 30)
        m = ctl.session.mask.array
        assert m[10:30, 10:30].all() and m.sum() == 255 * 400
        assert ctl.c.proposal is None and ctl.c.seed is None and not ctl.c.active
        assert ctl.accepted == [(10, 10, 30, 30)] and ctl.msgs[-1] == t("select.accepted")
        assert ctl.c.accept() is None

    def test_cancel_drops_proposal_and_a_running_request(self, ctl):
        ctl.prov.gate = threading.Event()
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: len(ctl.prov.calls) == 1)
        assert ctl.c.cancel() is True
        ctl.prov.gate.set()
        time.sleep(0.1)
        QApplication.processEvents()
        assert ctl.c.proposal is None and not ctl.c.active and ctl.c.seed is None
        assert ctl.msgs[-1] == t("select.cancelled") and ctl.c.cancel() is False

    def test_busy_indicator_only_after_200ms(self, ctl):
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: ctl.c.proposal is not None)
        assert ctl.busy == []
        ctl.prov.delay = 0.45
        ctl.c.press(QPointF(6, 6))
        assert wait_until(lambda: ctl.busy == [True])
        assert wait_until(lambda: ctl.busy == [True, False])

    def test_image_change_drops_the_proposal_and_a_new_context_is_built_for_the_new_version(self, ctl):
        ctl.prov.result = box_prop()
        ctl.c.press(QPointF(5, 5))
        assert wait_until(lambda: ctl.c.proposal is not None)
        keys_before = list(ctl.c._service.context_keys)
        ctl.c.on_image_changed()
        assert ctl.c.proposal is None and ctl.c.seed is None and not ctl.c.active
        ctl.session.history.push if False else None
        ctl.c.bind(Session(image_from_array(rnd(50, 50, 2))))
        assert ctl.c._service.context_keys == [] and keys_before


# ------------------------------------------------------------------------------------------------ חלון שלם (שרביט אמיתי)
class TestWindowWand:
    def test_actions_and_panel_default_to_brush_and_are_disabled_without_an_image(self, ew_noengine):
        w = ew_noengine.w
        assert w.select_panel.tool == "brush" and w.canvas.select_mode is None
        assert not w.act_tool_wand.isEnabled() and not w.act_select_accept.isEnabled()
        assert w.select_panel.tolerance == 20 and w.wand_ctl.tolerance == 20

    def test_actions_enable_with_an_image(self, ew_noengine):
        install(ew_noengine, two_halves())
        w = ew_noengine.w
        assert w.act_tool_wand.isEnabled() and not w.act_select_accept.isEnabled()

    def test_choosing_the_wand_sets_canvas_mode_and_radio(self, ew):
        install(ew, two_halves())
        wand(ew)
        assert ew.canvas.select_mode == "wand" and ew.w.select_panel.tool == "wand"
        assert ew.w.select_panel.radios["wand"].isChecked() and ew.w.select_panel.slider.isEnabled()

    def test_click_shows_a_proposal_and_touches_nothing_else(self, ew):
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew, count=150 * 200, rect=(0, 0, 150, 200))
        assert ew.canvas._proposal is ctl_of(ew).proposal and ew.w.act_select_accept.isEnabled()
        assert ew.w.select_panel.btn_accept.isEnabled() and ew.w.select_panel.btn_cancel.isEnabled()
        assert s.mask.empty and not s.modified and not s.history.can_undo and not s.busy
        assert not ew.w.mask_panel.history.can_undo and not ew.c.stroking

    def test_proposal_is_green_over_the_image_and_seed_ring_follows_the_click(self, ew):
        install(ew, two_halves())
        wand(ew)
        before = grab_rgb(ew.canvas)[:200, :300].astype(int)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        ew.canvas._hover = None
        QApplication.processEvents()
        after = grab_rgb(ew.canvas)[:200, :300].astype(int)
        px_b, px_a = before[150, 30], after[150, 30]                 # בתוך ההצעה, הרחק מטבעת הזרע
        assert px_a[1] > px_b[1] and px_a[0] < px_b[0] and px_a[2] < px_b[2]
        assert np.array_equal(before[:, 160:], after[:, 160:])        # מחוץ להצעה: זהה
        assert (ew.canvas._wand_seed.x(), ew.canvas._wand_seed.y()) == (50.0, 100.0)
        ctl_of(ew).cancel()
        QApplication.processEvents()
        ew.canvas._hover = None
        assert ew.canvas._wand_seed is None
        assert np.array_equal(grab_rgb(ew.canvas)[:200, :300].astype(int), before)

    def test_enter_accepts_and_removes_with_an_engine(self, ew):
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        key(ew, Qt.Key_Return)
        assert ew.idle() and s.modified and s.mask.empty
        assert tuple(s.rgb[100, 50]) == tuple(FILL) and tuple(s.rgb[100, 250]) == (200, 200, 200)
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None

    def test_enter_adds_to_mask_as_one_undo_step_without_an_engine(self, ew_noengine):
        ew = ew_noengine
        s = install(ew, two_halves())
        before = s.rgb.copy()
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        key(ew, Qt.Key_Return)
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None
        assert s.mask.array[:, :150].all() and not s.mask.array[:, 150:].any()
        assert ew.w.mask_panel.history.undo_count == 1 and ew.w.act_undo.isEnabled()
        assert not s.busy and not s.modified and np.array_equal(s.rgb, before)
        assert status(ew) == t("select.accepted")
        key(ew, Qt.Key_Z, Qt.ControlModifier)                          # Ctrl+Z מבטל קודם את עריכת המסכה הממתינה
        assert s.mask.empty
    def test_accepting_the_same_area_twice_reports_nothing_new(self, ew_noengine):
        ew = ew_noengine
        install(ew, two_halves())
        wand(ew)
        for _ in range(2):
            click(ew.canvas, 50, 100)
            assert proposal_is(ew)
            key(ew, Qt.Key_Return)
        assert status(ew) == t("select.nothing_new") and ew.w.mask_panel.history.undo_count == 1

    def test_accept_button_in_the_panel(self, ew_noengine):
        ew = ew_noengine
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 250, 100)
        assert proposal_is(ew)
        ew.w.select_panel.btn_accept.click()
        assert s.mask.array[:, 150:].all() and not s.mask.array[:, :150].any() and ctl_of(ew).proposal is None

    def test_cancel_button_in_the_panel(self, ew):
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        ew.w.select_panel.btn_cancel.click()
        assert ctl_of(ew).proposal is None and s.mask.empty and ew.canvas._proposal is None

    def test_a_click_after_the_removal_builds_a_context_for_the_new_image(self, ew):
        s = install(ew, bands())
        wand(ew)
        ew.w.wand_ctl.set_tolerance(0)
        click(ew.canvas, 10, 100)
        assert proposal_is(ew, count=30 * 200)
        keys1 = list(ctl_of(ew)._service.context_keys)
        key(ew, Qt.Key_Return)                                        # אישור = הסרה
        assert ew.idle() and tuple(s.rgb[100, 10]) == tuple(FILL)
        click(ew.canvas, 10, 100)
        assert proposal_is(ew)
        keys2 = list(ctl_of(ew)._service.context_keys)
        assert len(keys1) == 1 and len(keys2) == 2 and keys2[0] != keys2[1]

    def test_escape_cancels_the_proposal_first_then_jobs(self, ew):
        install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        assert ew.w.act_cancel.isEnabled()
        key(ew, Qt.Key_Escape)
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None and status(ew) == t("select.cancelled")
        assert not ew.w.act_select_accept.isEnabled()
        key(ew, Qt.Key_Escape)                                         # אין מה לבטל: לא קורס
        assert ctl_of(ew).proposal is None

    def test_new_click_replaces_the_proposal(self, ew):
        install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew, rect=(0, 0, 150, 200))
        click(ew.canvas, 250, 100)
        assert proposal_is(ew, rect=(150, 0, 300, 200))
        assert (ctl_of(ew).seed.x(), ctl_of(ew).seed.y()) == (250.0, 100.0)

    def test_click_outside_the_image_does_nothing(self, ew):
        install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 320, 100)
        time.sleep(0.05)
        QApplication.processEvents()
        assert ctl_of(ew).proposal is None and not ctl_of(ew).active

    def test_drag_changes_tolerance_live_and_syncs_the_panel(self, ew):
        install(ew, bands())
        wand(ew)
        c = ew.canvas
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(10, 100))
        assert proposal_is(ew, count=4 * 30 * 200)                    # סבילות 20: הפסים 100,106,112,118
        QTest.mouseMove(c, QPoint(10 + 10 * DRAG_PX_PER_STEP, 100))   # +10 → 30
        assert proposal_is(ew, count=6 * 30 * 200)                    # 100..130
        assert ew.w.select_panel.tolerance == 30 and ew.w.select_panel.spin.value() == 30
        QTest.mouseMove(c, QPoint(10 + 5 * DRAG_PX_PER_STEP, 100))    # +5 → 25
        assert proposal_is(ew, count=5 * 30 * 200)
        QTest.mouseMove(c, QPoint(10 - 4 * DRAG_PX_PER_STEP, 100))    # -4 → 16
        assert proposal_is(ew, count=3 * 30 * 200)                    # 100,106,112
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(10 - 4 * DRAG_PX_PER_STEP, 100))
        assert ctl_of(ew).tolerance == 16 and not c.gesture_active

    def test_tolerance_persists_into_the_next_click(self, ew):
        install(ew, bands())
        wand(ew)
        ctl_of(ew).set_tolerance(30)
        assert ew.w.select_panel.tolerance == 30
        click(ew.canvas, 10, 100)
        assert proposal_is(ew, count=6 * 30 * 200)

    def test_slider_and_spinner_change_tolerance_and_update_an_open_proposal(self, ew):
        install(ew, bands())
        wand(ew)
        click(ew.canvas, 10, 100)
        assert proposal_is(ew, count=4 * 30 * 200)
        ew.w.select_panel.slider.setValue(30)
        assert ctl_of(ew).tolerance == 30 and ew.w.select_panel.spin.value() == 30
        assert proposal_is(ew, count=6 * 30 * 200)
        ew.w.select_panel.spin.setValue(0)
        assert proposal_is(ew, count=30 * 200)                         # רק הפס הראשון

    def test_drag_clamps_at_the_slider_range(self, ew):
        install(ew, bands())
        wand(ew)
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(10, 100))
        QTest.mouseMove(ew.canvas, QPoint(290, 100))                  # +280px = +70 → נחתך ל-50
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(290, 100))
        assert ctl_of(ew).tolerance == MAX_UI_TOLERANCE and ew.w.select_panel.tolerance == MAX_UI_TOLERANCE

    def test_wand_drag_never_paints_the_mask_or_starts_jobs(self, ew):
        s = install(ew, two_halves())
        wand(ew)
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 20, 20))
        QTest.mouseMove(ew.canvas, scr(ew.canvas, 60, 20))
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 60, 20))
        assert s.mask.empty and not s.busy and not ew.c.stroking and ew.c.waiting is None

    def test_a_drag_after_a_proposal_changes_tolerance_not_a_scribble(self, ew):
        """עם הצעה פתוחה, ב-GrabCut גרירה היא משיכה מתקנת; בשרביט היא סבילות (הקנבס אינו במצב scribble)."""
        install(ew, bands())
        wand(ew)
        click(ew.canvas, 10, 100)
        assert proposal_is(ew)
        assert ew.canvas.select_scribble is False
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(150, 100))
        QTest.mouseMove(ew.canvas, QPoint(150 + 4 * DRAG_PX_PER_STEP, 100))
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(150 + 4 * DRAG_PX_PER_STEP, 100))
        assert ctl_of(ew).tolerance == 24 and ew.w.select_ctl.proposal is None

    def test_switching_to_brush_cancels_the_proposal_and_brush_works_again(self, ew):
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        ew.w.act_tool_brush.trigger()
        assert ctl_of(ew).proposal is None and ew.canvas.select_mode is None and ew.canvas._proposal is None
        QTest.mouseMove(ew.canvas, scr(ew.canvas, 100, 100))
        before = s.rgb.copy()
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        assert ew.idle() and not np.array_equal(s.rgb, before)

    def test_tool_shortcuts_W_and_P(self, ew):
        install(ew, two_halves())
        key(ew, Qt.Key_W)
        assert ew.canvas.select_mode == "wand" and ew.w.select_panel.tool == "wand"
        key(ew, Qt.Key_P)
        assert ew.canvas.select_mode is None and ew.w.select_panel.tool == "brush"

    def test_undo_of_a_commit_cancels_a_pending_proposal(self, ew):
        s = install(ew, two_halves())
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        assert ew.idle() and s.history.can_undo
        wand(ew)
        click(ew.canvas, 20, 20)
        assert proposal_is(ew)
        key(ew, Qt.Key_Z, Qt.ControlModifier)
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None

    def test_new_image_drops_the_proposal_and_the_next_click_uses_it(self, ew):
        install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        install(ew, two_halves(a=60, b=240))
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None and ctl_of(ew).seed is None
        click(ew.canvas, 50, 100)
        assert proposal_is(ew, rect=(0, 0, 150, 200))

    def test_no_wand_while_comparing(self, ew):
        install(ew, two_halves())
        wand(ew)
        ew.w.act_compare.setChecked(True)
        QApplication.processEvents()
        click(ew.canvas, 50, 100)
        time.sleep(0.05)
        QApplication.processEvents()
        assert ctl_of(ew).proposal is None and not ctl_of(ew).active
        ew.w.act_compare.setChecked(False)

    def test_wand_works_without_an_engine(self, ew_noengine):
        s = install(ew_noengine, two_halves())
        wand(ew_noengine)
        click(ew_noengine.canvas, 50, 100)
        assert proposal_is(ew_noengine)
        key(ew_noengine, Qt.Key_Return)                                # אין מנוע: המסכה נשארת מצוירת, בלי קריסה
        assert s.mask.array[:, :150].all() and not s.modified

    def test_closing_the_window_stops_the_wand_service(self, app):
        h = EWin(engine=False)
        install(h, two_halves())
        wand(h)
        click(h.canvas, 50, 100)
        assert proposal_is(h)
        thread = h.w.wand_ctl._service._thread
        h.close()
        assert wait_until(lambda: not thread.is_alive())


class TestCoexistenceWithSmartSelect:
    def test_switching_between_wand_and_box_cancels_the_other_tools_proposal(self, ew):
        install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        ew.w.select_panel.set_tool("box")
        assert ctl_of(ew).proposal is None and ew.canvas._proposal is None and ew.canvas.select_mode == "box"
        ew.w.select_panel.set_tool("wand")
        assert ew.canvas.select_mode == "wand" and not ew.w.select_ctl.active

    def test_enter_accepts_only_the_active_tools_proposal(self, ew_noengine):
        ew = ew_noengine
        s = install(ew, two_halves())
        wand(ew)
        click(ew.canvas, 250, 100)
        assert proposal_is(ew)
        assert not ew.w.select_ctl.has_proposal
        key(ew, Qt.Key_Return)
        assert s.mask.array[:, 150:].all() and not s.mask.array[:, :150].any()

    def test_box_tool_still_gets_the_scribble_mode_when_it_has_input(self, ew):
        install(ew, two_halves())
        ew.w.select_panel.set_tool("box")
        assert ew.canvas.select_scribble is False
        ew.w.select_ctl.request_box((20, 20, 120, 120))
        assert wait_until(lambda: ew.w.select_ctl.has_input)
        assert wait_until(lambda: ew.canvas.select_scribble is True)

    def test_escape_cancels_whichever_tool_has_state(self, ew):
        install(ew, two_halves())
        ew.w.select_panel.set_tool("box")
        ew.w.select_ctl.request_box((20, 20, 120, 120))
        assert wait_until(lambda: ew.w.select_ctl.has_input)
        key(ew, Qt.Key_Escape)
        assert not ew.w.select_ctl.active
        wand(ew)
        click(ew.canvas, 50, 100)
        assert proposal_is(ew)
        key(ew, Qt.Key_Escape)
        assert not ctl_of(ew).active

    def test_image_change_cancels_both_controllers(self, ew):
        s = install(ew, two_halves())
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 100, 100))
        assert ew.idle() and s.history.can_undo
        ew.w.select_panel.set_tool("box")
        ew.w.select_ctl.request_box((20, 20, 120, 120))
        assert wait_until(lambda: ew.w.select_ctl.has_input)
        key(ew, Qt.Key_Z, Qt.ControlModifier)
        assert not ew.w.select_ctl.active and not ctl_of(ew).active
