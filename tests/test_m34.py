"""M3.4: מברשת ולולאת Inpaint בממשק (PARITY-SPEC §3). Qt על פלטפורמת offscreen, Scheduler אמיתי ומודל מדומה.

הבדיקות מוכיחות את הלולאה (משיכה -> מסכה -> Job -> Commit -> תצוגה), לא את איכות ההסרה (מודל סינתטי) ולא את מראה
הממשק בעין. מה שלא נבדק כאן: Windows, מקלדת עברית (קיצורי `[` `]` `E`), i5-6500T, Big LaMa אמיתי."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic import strings  # noqa: E402
from fixpic.engine import JobScheduler  # noqa: E402
from fixpic.errors import ModelLoadError  # noqa: E402
from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.pipeline import Pipeline, PipelineConfig  # noqa: E402
from fixpic.session import HistoryAction, MaskLayer, Session  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from fixpic.ui.brush_panel import BrushPanel, brush_range_for  # noqa: E402
from fixpic.ui.canvas import OVERLAY_OPACITY, PENDING_COLOR, Canvas  # noqa: E402
from fixpic.ui.engine import default_engine_factory  # noqa: E402
from fixpic.ui.edit_loop import EngineState  # noqa: E402
from fixpic.ui.main_window import MainWindow  # noqa: E402
from test_backends import tiny  # noqa: E402,F401  (fixture)
from test_engine import FILL, FakeModel, rnd  # noqa: E402
from test_scheduler import GateModel  # noqa: E402

T = 10.0


@pytest.fixture(scope="module")
def app():
    return create_app([])


def wait_until(cond, timeout=T):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return bool(cond())


def scr(canvas, x, y) -> QPoint:
    """מרכז פיקסל תמונה (x, y) על המסך."""
    p = canvas.image_to_view(QPointF(x + 0.5, y + 0.5))
    return QPoint(round(p.x()), round(p.y()))


def drag(canvas, pts, button=Qt.LeftButton):
    QTest.mousePress(canvas, button, Qt.NoModifier, scr(canvas, *pts[0]))
    for p in pts[1:]:
        QTest.mouseMove(canvas, scr(canvas, *p))
    QTest.mouseRelease(canvas, button, Qt.NoModifier, scr(canvas, *pts[-1]))


def grab_rgb(widget) -> np.ndarray:
    img = widget.grab().toImage().convertToFormat(widget.grab().toImage().Format.Format_RGB888)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3]
    return arr.reshape(img.height(), img.width(), 3).copy()


# ---------------------------------------------------------------------------- Session.cancel_pending
def make_session(model, h=120, w=120):
    sched = JobScheduler(Pipeline(model, PipelineConfig())).start(warmup=False)
    s = Session(image_from_array(np.full((h, w, 3), 100, np.uint8)), sched)
    return s, sched


def test_cancel_pending_idle_returns_false_and_never_undoes():
    s, sched = make_session(GateModel(gated=False))
    try:
        s.mask.paint([(60, 60)], 20)
        job = s.submit_stroke()
        assert wait_until(lambda: job.finished)
        assert s.finish(job).patch is not None
        committed = s.rgb.copy()
        assert s.cancel_pending() is False
        assert np.array_equal(s.rgb, committed) and s.history.peek_undo() is not None   # ה-Commit לא בוטל
    finally:
        sched.shutdown()


def test_cancel_pending_discards_running_and_queued_without_touching_history():
    model = GateModel()
    s, sched = make_session(model)
    try:
        before = s.rgb.copy()
        jobs = []
        for x in (30, 60, 90):
            s.mask.paint([(x, 60)], 10)
            jobs.append(s.submit_stroke())
        assert model.entered.acquire(timeout=T)                  # הראשונה רצה, שתיים בתור
        assert s.busy and s.cancel_pending() is True and not s.busy
        model.gate.release(5)
        assert wait_until(lambda: all(j.finished for j in jobs) and sched.current is None)
        assert all(s.finish(j).kind.value == "stale" for j in jobs)
        assert s.mask.empty and np.array_equal(s.rgb, before) and not s.modified   # נזרקו, לא הוחזרו לשכבה
        assert s.undo() is HistoryAction.NOTHING                                   # וגם אין מה לבטל
    finally:
        model.gate.release(5)
        sched.shutdown()


# ---------------------------------------------------------------------------- BrushPanel
class TestBrushPanel:
    def test_range_scales_with_image(self):
        assert brush_range_for(1000, 800) == (24, 200)
        assert brush_range_for(6000, 4000) == (120, 1000)
        assert brush_range_for(20, 20) == (8, 32)                  # תמונה זעירה: מינימום סביר
        assert brush_range_for(100000, 100000) == (2000, 2000)     # תקרה
        for hw in ((10, 10), (500, 40), (4000, 6000)):
            d, top = brush_range_for(*hw)
            assert 1 <= d <= top

    def test_slider_spin_stay_in_sync_and_emit_once(self, app):
        p = BrushPanel()
        got = []
        p.diameterChanged.connect(got.append)
        p.configure_for_image(1000, 800)
        assert p.diameter == 24 and p.slider.value() == 24 and p.slider.maximum() == 200 and got == [24]
        p.slider.setValue(50)
        assert p.spin.value() == 50 and got[-1] == 50
        p.spin.setValue(7)
        assert p.slider.value() == 7 and got[-1] == 7
        p.set_diameter(10_000)
        assert p.diameter == 200                                   # נחתך לטווח

    def test_scale_always_moves_at_least_one_and_respects_bounds(self, app):
        p = BrushPanel()
        p.configure_for_image(1000, 800)
        p.set_diameter(3)
        p.scale_diameter(1.2)
        assert p.diameter == 4                                     # 3.6 -> 4
        p.set_diameter(1)
        p.scale_diameter(1 / 1.2)
        assert p.diameter == 1                                     # לא יורד מתחת למינימום
        p.set_diameter(200)
        p.scale_diameter(1.2)
        assert p.diameter == 200
        p.set_diameter(5)
        p.scale_diameter(1.01)
        assert p.diameter == 6                                     # לפחות פיקסל

    def test_erase_signal(self, app):
        p = BrushPanel()
        got = []
        p.eraseChanged.connect(got.append)
        p.set_erase(True)
        assert p.erase and got == [True]


# ---------------------------------------------------------------------------- Canvas: משיכות ושכבת מסכה
_CANVASES: list[Canvas] = []


@pytest.fixture(autouse=True)
def _close_canvases():
    """`QTest.mouseMove` על QWidget הוא `QCursor::setPos`: האירוע מגיע לחלון העליון שמתחת לעכבר. קנבסים שנשארו פתוחים
    מבדיקה קודמת היו גונבים אותו, ולכן סוגרים כל קנבס בסוף הבדיקה."""
    yield
    for c in _CANVASES:
        c.close()
        c.deleteLater()
    _CANVASES.clear()
    QApplication.processEvents()


def painted_canvas(h=100, w=200, value=100):
    c = Canvas()
    _CANVASES.append(c)
    c.resize(800, 600)
    c.show()
    c.set_image(np.full((h, w, 3), value, np.uint8))
    layer = MaskLayer(h, w)
    c.set_mask_layer(layer)
    c.set_painting(True)
    QApplication.processEvents()
    return c, layer


class TestCanvasStrokes:
    def test_left_draws_right_erases_and_coordinates_are_pixel_centers(self, app):
        c, _ = painted_canvas()
        c.scale, c.offset = 4.0, QPointF(10, 5)
        got = []
        c.strokeStarted.connect(lambda p, e: got.append(("start", p.x(), p.y(), e)))
        c.strokeMoved.connect(lambda p: got.append(("move", p.x(), p.y())))
        c.strokeEnded.connect(lambda: got.append(("end",)))
        # מסך (10+4*10+1, 5+4*20+1) = בתוך פיקסל (10, 20): מרכזו (10, 20)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(10 + 41, 5 + 81))
        QTest.mouseMove(c, QPoint(10 + 4 * 12 + 1, 5 + 81))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(10 + 4 * 12 + 1, 5 + 81))
        assert got[0][0] == "start" and got[0][3] is False
        assert got[0][1] == pytest.approx(10.25 - 0.5) and got[0][2] == pytest.approx(20.25 - 0.5)
        assert round(got[0][1]) == 10 and round(got[0][2]) == 20           # אותו פיקסל שמתחת לעכבר
        assert got[1][0] == "move" and round(got[1][1]) == 12 and got[-1] == ("end",)
        got.clear()
        QTest.mousePress(c, Qt.RightButton, Qt.NoModifier, QPoint(60, 60))
        QTest.mouseRelease(c, Qt.RightButton, Qt.NoModifier, QPoint(60, 60))
        assert got[0][0] == "start" and got[0][3] is True and got[-1] == ("end",)

    def test_erase_mode_makes_left_erase(self, app):
        c, _ = painted_canvas()
        c.erase_mode = True
        got = []
        c.strokeStarted.connect(lambda p, e: got.append(e))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        assert got == [True]

    def test_left_drag_pans_only_outside_painting_mode(self, app):
        c, _ = painted_canvas()
        c.scale = 2.0
        c.offset = QPointF(0, 0)
        o = QPointF(c.offset)
        starts = []
        c.strokeStarted.connect(lambda p, e: starts.append(1))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(340, 330))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(340, 330))
        assert starts == [1] and c.offset == o                              # צייר, לא הזיז
        QTest.mousePress(c, Qt.MiddleButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(320, 310))
        QTest.mouseRelease(c, Qt.MiddleButton, Qt.NoModifier, QPoint(320, 310))
        assert c.offset != o and starts == [1]                              # אמצעי מזיז
        c.set_painting(False)
        o2 = QPointF(c.offset)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(310, 300))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(310, 300))
        assert c.offset != o2 and starts == [1]                             # מחוץ למצב ציור: שמאלי מזיז

    def test_space_pans_instead_of_drawing(self, app):
        c, _ = painted_canvas()
        c.scale = 2.0
        c.offset = QPointF(0, 0)
        o = QPointF(c.offset)
        starts = []
        c.strokeStarted.connect(lambda p, e: starts.append(1))
        QTest.keyPress(c, Qt.Key_Space)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(320, 300))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(320, 300))
        QTest.keyRelease(c, Qt.Key_Space)
        assert starts == [] and c.offset != o

    def test_turning_painting_off_mid_stroke_ends_it(self, app):
        c, _ = painted_canvas()
        ended = []
        c.strokeEnded.connect(lambda: ended.append(1))
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        c.set_painting(False)
        assert ended == [1]
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(100, 100))
        assert ended == [1]                                                 # שחרור מאוחר לא משדר שוב


class TestCanvasOverlay:
    def test_mask_tints_exactly_the_mask_pixels_at_1_to_1(self, app):
        c, layer = painted_canvas(value=100)
        c.scale, c.offset = 1.0, QPointF(0, 0)
        c._fit_mode = False
        layer.paint([(50, 40), (90, 40)], 10)
        c.invalidate_overlay()
        shot = grab_rgb(c)
        inside, outside = shot[40, 70], shot[80, 70]
        assert tuple(outside) == (100, 100, 100)                            # מחוץ למסכה: התמונה כמות שהיא
        assert inside[0] > 100 and inside[1] < 100 and inside[2] < 100      # גוון אדום
        m = np.asarray(layer.array) > 0
        tinted = (shot[:100, :200] != 100).any(axis=2)
        assert np.array_equal(tinted, m)                                    # כל פיקסל מסכה, ורק הוא, צבוע

    def test_alignment_when_zoomed_in_and_out(self, app):
        for scale, off in ((4.0, QPointF(10, 5)), (0.5, QPointF(30, 20)), (1.7, QPointF(-20, 7))):
            c, layer = painted_canvas(h=100, w=200, value=100)
            c.scale, c.offset = scale, off
            c._fit_mode = False
            win = np.zeros((100, 200), np.uint8)
            win[30:50, 60:100] = 255
            layer.add_window((60, 30, 100, 50), win[30:50, 60:100])
            c.invalidate_overlay()
            shot = grab_rgb(c)
            cx = int(off.x() + 80 * scale)
            cy = int(off.y() + 40 * scale)
            assert shot[cy, cx, 0] > 100 > shot[cy, cx, 1], f"פנים המסכה לא נצבע ב-{scale}"
            ox = int(off.x() + 30 * scale)
            assert tuple(shot[cy, ox]) == (100, 100, 100), f"מחוץ למסכה נצבע ב-{scale}"
            # הקצה: פיקסל מסך אחד לפני תחילת המסכה אינו צבוע מעבר לשבריר ההקטנה
            ex = int(off.x() + 60 * scale)
            assert shot[cy, ex - 2 if scale >= 1 else ex - 1, 1] == 100

    def test_pending_masks_draw_blue_and_current_red(self, app):
        c, layer = painted_canvas(value=100)
        c.scale, c.offset = 1.0, QPointF(0, 0)
        c._fit_mode = False
        pend = np.zeros((100, 200), np.uint8)
        pend[10:30, 10:30] = 255
        c.set_pending_source(lambda: [pend])
        layer.paint([(150, 50)], 20)
        c.invalidate_overlay()
        shot = grab_rgb(c)
        p, m = shot[20, 20], shot[50, 150]
        assert p[2] > 100 > p[0] and m[0] > 100 > m[2]

    def test_overlay_is_cached_across_hover_and_recomputed_on_change(self, app, monkeypatch):
        c, layer = painted_canvas()
        c.scale, c.offset = 1.0, QPointF(0, 0)
        c._fit_mode = False
        layer.paint([(50, 40)], 12)
        calls = []
        orig = c._render_overlay
        monkeypatch.setattr(c, "_render_overlay", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
        c.invalidate_overlay()
        c.grab()
        n = len(calls)
        assert n >= 1
        for x in (100, 110, 120):
            QTest.mouseMove(c, QPoint(x, 100))
            c.grab()
        assert len(calls) == n                                              # ריחוף לא מחשב מחדש
        layer.paint([(80, 40)], 12)
        c.grab()
        assert len(calls) > n                                               # שינוי מסכה כן
        n2 = len(calls)
        c.pan_by(QPointF(5, 5))
        c.grab()
        assert len(calls) > n2                                              # וכן הזזה

    def test_empty_layer_and_no_pending_skip_overlay_work(self, app, monkeypatch):
        c, layer = painted_canvas()
        calls = []
        monkeypatch.setattr(c, "_render_overlay", lambda *a, **k: calls.append(1))
        c.grab()
        assert calls == []

    def test_brush_circle_follows_mouse_and_scales_with_zoom(self, app):
        c, _ = painted_canvas(value=100)
        c.scale, c.offset = 2.0, QPointF(0, 0)
        c._fit_mode = False
        c.set_brush_diameter(40)                                            # רדיוס מסך 40
        QTest.mouseMove(c, QPoint(200, 150))
        shot = grab_rgb(c)
        ring = shot[150, 240]                                               # על הטבעת: לבן או שחור, לא האפור 100
        assert tuple(ring) != (100, 100, 100)
        assert tuple(shot[150, 330]) == (100, 100, 100)                     # רחוק מהטבעת: ללא שינוי
        c.set_painting(False)
        assert tuple(grab_rgb(c)[150, 240]) == (100, 100, 100)              # מחוץ למצב ציור אין עיגול


# ---------------------------------------------------------------------------- MainWindow + מנוע
class EWin:
    """MainWindow עם Scheduler אמיתי ומודל מדומה; דיאלוגי שגיאה נרשמים."""

    def __init__(self, model=None, engine=True, factory=None, big=False):
        self.model = model if model is not None else FakeModel()
        self.sched = JobScheduler(Pipeline(self.model, PipelineConfig()))
        if engine and factory is None:
            factory = lambda: self.sched      # noqa: E731
        self.w = MainWindow(engine_factory=factory if engine else None)
        if big:
            # MainWindow מתאים את גודלו למסך (M3.10); במסך offscreen קטן (למשל runner של Windows) ה-Canvas יצא צר מ-300px
            # והבדיקות שמצלמות אזור 300x300 נכשלו. גודל מינימלי קבוע = אותו Canvas בכל מסך (רק בפיקסטורות; בדיקות
            # M3.10 בודקות במפורש את ההתאמה למסך קטן).
            self.w.setMinimumSize(max(self.w.minimumWidth(), 736), max(self.w.minimumHeight(), 736))
            self.w.resize(max(self.w.width(), 736), max(self.w.height(), 736))
        self.errors: list[tuple[str, dict]] = []
        self.w._error = lambda key, **f: self.errors.append((key, f))
        self.w.show()

    @property
    def c(self):
        return self.w.controller

    @property
    def s(self) -> Session:
        return self.w.session

    @property
    def canvas(self) -> Canvas:
        return self.w.canvas

    def ready(self, timeout=T) -> bool:
        return wait_until(lambda: self.c.state is EngineState.READY, timeout)

    def install(self, h=300, w=300, seed=0, diameter=20) -> Session:
        self.w._install(image_from_array(rnd(h, w, seed)), "x.png")
        self.w.brush_panel.set_diameter(diameter)
        QApplication.processEvents()
        return self.s

    def idle(self, timeout=T) -> bool:
        return wait_until(lambda: self.s is not None and not self.s.busy and self.c.waiting is None, timeout)

    def close(self):
        if isinstance(self.model, GateModel):
            self.model.gate.release(20)
        self.w._runner.wait_idle()
        self.c.shutdown()
        self.w.session = None
        self.w.close()


@pytest.fixture()
def ew(app):
    h = EWin(big=True)
    assert h.ready()
    yield h
    h.close()


@pytest.fixture()
def ew_gate(app):
    h = EWin(model=GateModel(), big=True)
    assert h.ready()
    yield h
    h.close()


@pytest.fixture()
def ew_noengine(app):
    h = EWin(engine=False, big=True)
    yield h
    h.close()


class TestEditLoop:
    def test_stroke_release_runs_job_commits_and_refreshes_canvas(self, ew):
        s = ew.install()
        before = s.rgb.copy()
        drag(ew.canvas, [(100, 100), (120, 100), (140, 100)])
        assert ew.idle() and s.modified
        assert not np.array_equal(s.rgb, before)
        assert tuple(s.rgb[100, 120]) == tuple(FILL)                       # מרכז המשיכה מולא
        far = np.ones((300, 300), bool)
        far[60:140, 80:180] = False
        assert np.array_equal(s.rgb[far], before[far])                      # הרחק מהמשיכה: זהה לבייט
        assert s.mask.empty                                                 # השכבה התרוקנה בהגשה
        assert np.array_equal(ew.canvas.pyr.arrays[0], s.rgb)               # ה-Canvas מציג את התוצאה
        assert "*" in ew.w.windowTitle()
        assert s.history.peek_undo() is not None                            # Patch ל-Undo נשמר

    def test_second_stroke_runs_on_updated_image(self, ew):
        s = ew.install()
        drag(ew.canvas, [(60, 60), (80, 60)])
        assert ew.idle()
        first = s.rgb.copy()
        drag(ew.canvas, [(200, 200), (220, 200)])
        assert ew.idle()
        assert tuple(s.rgb[60, 70]) == tuple(FILL) and tuple(s.rgb[200, 210]) == tuple(FILL)
        assert np.array_equal(s.rgb[:100, :], first[:100, :])               # המשיכה השנייה לא נגעה בראשונה

    def test_right_button_and_erase_mode_erase_without_running_a_job(self, ew_gate):
        ew = ew_gate
        s = ew.install()
        s.mask.paint([(100, 100), (160, 100)], 20)
        assert not s.mask.empty
        drag(ew.canvas, [(100, 100), (160, 100)], button=Qt.RightButton)
        assert s.mask.empty and not s.busy and ew.model.log == []           # נמחק, ללא Job
        s.mask.paint([(100, 100), (160, 100)], 20)
        ew.w.brush_panel.set_erase(True)
        assert ew.canvas.erase_mode and ew.c.erase_mode
        drag(ew.canvas, [(100, 100), (160, 100)])
        assert s.mask.empty and ew.model.log == []

    def test_brush_size_drives_mask_and_job_brush_px(self, ew_noengine):
        ew = ew_noengine
        s = ew.install(diameter=30)
        drag(ew.canvas, [(100, 100)])
        assert s.mask.brush_px == 30
        ys, xs = np.nonzero(np.asarray(s.mask.array))
        assert xs.min() >= 84 and xs.max() <= 116 and (xs.max() - xs.min()) >= 28   # קוטר ~30
        ew.w.act_brush_bigger.trigger()
        assert ew.w.brush_panel.diameter == 36 and ew.c.diameter == 36.0 and ew.canvas.brush_diameter == 36.0
        ew.w.act_brush_smaller.trigger()
        assert ew.w.brush_panel.diameter == 30

    def test_without_engine_strokes_only_paint_the_mask(self, ew_noengine):
        ew = ew_noengine
        s = ew.install()
        drag(ew.canvas, [(100, 100), (150, 100)])
        assert not s.mask.empty and not s.busy and ew.c.state is EngineState.NONE
        assert not s.modified

    def test_painting_disabled_while_loading_or_saving_and_without_image(self, ew_noengine):
        ew = ew_noengine
        assert ew.canvas.painting is False                                   # אין תמונה
        ew.install()
        assert ew.canvas.painting is True
        ew.w._set_busy(True)
        assert ew.canvas.painting is False
        ew.w._set_busy(False)
        assert ew.canvas.painting is True

    def test_queue_full_keeps_mask_then_submits_when_space_frees(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=300, w=500)
        xs = (40, 120, 200, 280, 360)
        drag(ew.canvas, [(xs[0], 100)])
        assert ew.model.entered.acquire(timeout=T)                           # הראשונה רצה
        for x in xs[1:4]:
            drag(ew.canvas, [(x, 100)])                                      # שלוש בתור
        assert ew.sched.pending == 3 and ew.c.waiting is None
        drag(ew.canvas, [(xs[4], 100)])                                      # הרביעית בתור: התור מלא
        assert ew.c.waiting == "queue" and not s.mask.empty and ew.sched.pending == 3
        assert "מלא" in ew.w.statusBar().currentMessage()
        ew.model.gate.release(1)                                             # הראשונה מסתיימת -> מתפנה מקום
        assert wait_until(lambda: ew.c.waiting is None)
        ew.model.gate.release(10)
        assert ew.idle()
        assert s.mask.empty
        for x in xs:
            assert tuple(s.rgb[100, x]) == (9, 9, 9), x                      # כל חמש המשיכות הוחלו

    def test_cancel_during_processing_discards_and_keeps_history(self, ew_gate):
        ew = ew_gate
        s = ew.install()
        drag(ew.canvas, [(60, 60)])
        assert ew.model.entered.acquire(timeout=T)
        ew.model.gate.release(1)
        assert ew.idle() and s.modified
        done = s.rgb.copy()
        drag(ew.canvas, [(200, 200)])
        assert ew.model.entered.acquire(timeout=T)
        assert ew.w.act_cancel.isEnabled()
        ew.w.act_cancel.trigger()
        assert not s.busy and s.mask.empty
        assert "בוטל" in ew.w.statusBar().currentMessage()
        ew.model.gate.release(5)
        assert wait_until(lambda: ew.sched.current is None)
        QApplication.processEvents()
        assert np.array_equal(s.rgb, done)                                   # הראשונה נשארה, השנייה לא הוחלה
        assert s.history.peek_undo() is not None and not ew.w.act_cancel.isEnabled()

    def test_cancel_drops_mask_waiting_for_queue_too(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=300, w=600)
        for x in (40, 120, 200, 280):
            drag(ew.canvas, [(x, 100)])
            if x == 40:
                assert ew.model.entered.acquire(timeout=T)
        drag(ew.canvas, [(360, 100)])
        drag(ew.canvas, [(440, 100)])
        assert ew.c.waiting == "queue"
        assert ew.c.cancel() is True
        assert ew.c.waiting is None and s.mask.empty and not s.busy

    def test_cancel_when_idle_is_a_noop(self, ew):
        s = ew.install()
        assert ew.c.cancel() is False and not s.modified

    def test_failure_keeps_mask_reports_once_and_does_not_retry(self, app):
        class Boom(FakeModel):
            calls = 0

            def run(self, image, mask, cancel=None):
                Boom.calls += 1
                raise RuntimeError("boom")
        h = EWin(model=Boom())
        try:
            assert h.ready()
            s = h.install()
            before = s.rgb.copy()
            drag(h.canvas, [(100, 100), (120, 100)])
            assert wait_until(lambda: any(k == "error.inpaint" for k, _ in h.errors))
            assert wait_until(lambda: not s.busy)
            time.sleep(0.3)                                                   # אין ניסיון חוזר אוטומטי
            QApplication.processEvents()
            assert Boom.calls == 1 and len([1 for k, _ in h.errors if k == "error.inpaint"]) == 1
            assert not s.mask.empty and np.array_equal(s.rgb, before) and not s.modified   # §3.6
            assert h.c.waiting is None
        finally:
            h.close()

    def test_indicator_only_after_200ms(self, ew_gate):
        ew = ew_gate
        s = ew.install()
        shown = []
        ew.c.processingChanged.connect(shown.append)
        drag(ew.canvas, [(100, 100)])
        assert ew.model.entered.acquire(timeout=T)
        assert s.busy and shown == [] and ew.w.proc_progress.isHidden()      # מיד אחרי ההגשה: ללא מחוון
        wait_until(lambda: False, 0.1)                                       # 100ms עם לולאת אירועים פעילה: עדיין לא
        assert s.busy and shown == [] and ew.w.proc_progress.isHidden()
        assert wait_until(lambda: shown == [True], 3)
        assert not ew.w.proc_progress.isHidden() and "מעבד" in ew.w.lbl_proc.text()
        ew.model.gate.release(1)
        assert ew.idle()
        assert shown == [True, False] and ew.w.proc_progress.isHidden() and ew.w.lbl_proc.text() == ""

    def test_fast_job_never_shows_indicator(self, ew):
        s = ew.install(h=120, w=160)
        shown = []
        ew.c.processingChanged.connect(shown.append)
        drag(ew.canvas, [(60, 60), (80, 60)])
        assert ew.idle() and s.modified
        assert shown == [] and ew.w.proc_progress.isHidden()

    def test_processing_text_counts_queued_strokes(self, ew_gate):
        ew = ew_gate
        ew.install(h=300, w=400)
        drag(ew.canvas, [(40, 100)])
        assert ew.model.entered.acquire(timeout=T)
        drag(ew.canvas, [(160, 100)])
        assert wait_until(lambda: ew.c.indicator_on, 3)
        assert "2" in ew.w.lbl_proc.text()

    def test_pending_stroke_is_drawn_blue_while_processing(self, ew_gate):
        ew = ew_gate
        s = ew.install()
        ew.canvas.set_brush_diameter(20)
        drag(ew.canvas, [(100, 100), (140, 100)])
        assert ew.model.entered.acquire(timeout=T)
        assert s.mask.empty and len(s.pending_masks) == 1
        ew.canvas.invalidate_overlay()
        shot_with = grab_rgb(ew.canvas)
        p = scr(ew.canvas, 120, 100)
        base = np.asarray(ew.canvas.pyr.arrays[0])[100, 120].astype(int)
        px = shot_with[p.y(), p.x()].astype(int)
        a = OVERLAY_OPACITY / 255
        expected = base + (np.array(PENDING_COLOR) - base) * a                # מיזוג הצבע הכחול מעל התמונה
        assert np.abs(px - expected).max() <= 3, (px, expected)

    def test_opening_another_image_cancels_old_jobs(self, ew_gate):
        ew = ew_gate
        old = ew.install(seed=1)
        drag(ew.canvas, [(100, 100)])
        assert ew.model.entered.acquire(timeout=T)
        new = ew.install(seed=2)
        assert new is not old and not old.busy and ew.s is new
        before = new.rgb.copy()
        ew.model.gate.release(5)
        assert wait_until(lambda: ew.sched.current is None)
        for _ in range(20):
            QApplication.processEvents()
            time.sleep(0.01)
        assert np.array_equal(new.rgb, before) and not new.modified and ew.errors == []

    def test_job_finishing_after_session_is_dropped_is_ignored(self, ew_gate):
        ew = ew_gate
        ew.install()
        drag(ew.canvas, [(100, 100)])
        assert ew.model.entered.acquire(timeout=T)
        ew.w.session = None
        ew.c.bind(None)
        ew.model.gate.release(1)
        assert wait_until(lambda: ew.sched.current is None)
        for _ in range(10):
            QApplication.processEvents()
        assert ew.errors == []


class TestEngineLifecycle:
    def test_engine_loads_in_background_and_window_stays_responsive(self, app):
        gate = threading.Event()
        holder = {}

        def factory():
            assert gate.wait(T)
            return holder["h"].sched
        h = EWin(factory=factory)
        holder["h"] = h
        try:
            assert h.c.state is EngineState.LOADING
            assert "טוען" in h.w.statusBar().currentMessage()
            h.install()                                                       # אפשר לעבוד בזמן הטעינה
            gate.set()
            assert h.ready()
            assert h.s.scheduler is h.sched                                   # ה-Session הקיים קיבל את המנוע
            assert "מוכן" in h.w.statusBar().currentMessage()
        finally:
            gate.set()
            h.close()

    def test_stroke_while_loading_waits_then_runs_when_ready(self, app):
        gate = threading.Event()
        holder = {}

        def factory():
            assert gate.wait(T)
            return holder["h"].sched
        h = EWin(factory=factory)
        holder["h"] = h
        try:
            s = h.install()
            drag(h.canvas, [(100, 100), (130, 100)])
            assert h.c.waiting == "engine" and not s.mask.empty and not s.busy
            assert "נטען" in h.w.statusBar().currentMessage()
            gate.set()
            assert wait_until(lambda: s.modified and h.c.waiting is None and not s.busy)
            assert s.mask.empty and tuple(s.rgb[100, 115]) == tuple(FILL)
        finally:
            gate.set()
            h.close()

    def test_default_factory_loads_a_real_model_and_the_whole_loop_runs(self, app, tiny, tmp_path):  # noqa: F811
        """הנתיב האמיתי: ModelManager -> AutoModel (OpenVINO/ORT) -> Pipeline -> Scheduler, דרך משיכה בממשק.
        המודל הקטן הופך צבעים בתוך המסכה (`255 - x`), ולכן אפשר לוודא מספרית שה-Commit הגיע מהמודל."""
        d = tmp_path / "models" / "tiny"
        d.mkdir(parents=True)
        shutil.copy(tiny, d / "w.onnx")
        (d / "model.json").write_text(json.dumps({"schema": 1, "name": "tiny", "file": "w.onnx"}), encoding="utf-8")
        h = EWin(factory=default_engine_factory(tmp_path / "models", "tiny", threads=2))
        try:
            assert h.ready(30)
            s = h.install()
            before = s.rgb.copy()
            drag(h.canvas, [(100, 100), (140, 100)])
            assert wait_until(lambda: s.modified and not s.busy, 30)
            inner = s.rgb[100, 120].astype(int)
            assert np.abs(inner - (255 - before[100, 120].astype(int))).max() <= 2
            assert np.array_equal(s.rgb[:50], before[:50])
            assert h.errors == []
        finally:
            h.close()

    def test_new_image_after_ready_gets_the_scheduler(self, ew):
        a = ew.install(seed=1)
        b = ew.install(seed=2)
        assert a.scheduler is ew.sched and b.scheduler is ew.sched

    def test_load_failure_reports_and_strokes_are_dropped_gracefully(self, app):
        def factory():
            raise ModelLoadError("אין מודל")
        h = EWin(factory=factory)
        try:
            assert wait_until(lambda: h.c.state is EngineState.FAILED)
            assert [k for k, _ in h.errors] == ["error.engine"] and "אין מודל" in h.errors[0][1]["detail"]
            s = h.install()
            drag(h.canvas, [(100, 100)])
            assert s.mask.empty and not s.busy                                # אין מה להגיש: נזרק
            assert strings.t("error.no_engine") in h.w.statusBar().currentMessage()
            drag(h.canvas, [(100, 100)])                                      # שוב: בלי קריסה ובלי דיאלוג נוסף
            assert [k for k, _ in h.errors] == ["error.engine"]
        finally:
            h.close()

    def test_unexpected_load_exception_is_reported_not_raised(self, app):
        def factory():
            raise ZeroDivisionError("x")
        h = EWin(factory=factory)
        try:
            assert wait_until(lambda: h.c.state is EngineState.FAILED)
            assert "ZeroDivisionError" in h.errors[0][1]["detail"]
        finally:
            h.close()

    def test_close_shuts_the_scheduler_down(self, app):
        h = EWin()
        assert h.ready()
        t = h.sched._thread
        assert t is not None and t.is_alive()
        h.w.close()
        assert wait_until(lambda: not t.is_alive(), 8)
        h.w.session = None

    def test_closing_with_a_running_job_does_not_hang_or_crash(self, app):
        h = EWin(model=GateModel())
        assert h.ready()
        h.install()
        drag(h.canvas, [(100, 100)])
        assert h.model.entered.acquire(timeout=T)
        h.model.gate.release(5)
        t0 = time.time()
        h.w.close()
        assert time.time() - t0 < 8
        h.w.session = None


class TestErrorDialogs:
    def test_nested_failures_show_a_single_dialog(self, app):
        h = EWin(engine=False)
        try:
            calls = []

            def fake_error(key, **f):
                calls.append(key)
                h.w._error_once("error.inpaint", detail="שנייה")             # נכשל בזמן שהדיאלוג פתוח
            h.w._error = fake_error
            h.w._error_once("error.inpaint", detail="ראשונה")
            assert calls == ["error.inpaint"]
            assert not h.w._error_open
        finally:
            h.close()
