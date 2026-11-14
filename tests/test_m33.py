"""M3.3: שלד ממשק (strings, Canvas עם Zoom/Pan, פתיחה/גרירה/הדבקה, שמירה ברקע). Qt על פלטפורמת offscreen."""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent, QGuiApplication, QImage, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic import strings  # noqa: E402
from fixpic.errors import InvalidInputError  # noqa: E402
from fixpic.imgio import image_from_array, load_image, save_image  # noqa: E402
from fixpic.pipeline import InpaintResult  # noqa: E402
from fixpic.pipeline.roi import RoiPlan  # noqa: E402
from fixpic.ui.app import create_app  # noqa: E402
from fixpic.ui.canvas import MAX_SCALE, MIN_SCALE, Canvas, Pyramid  # noqa: E402
from fixpic.ui.convert import qimage_to_array  # noqa: E402
from fixpic.ui.main_window import MainWindow  # noqa: E402
from fixpic.ui.workers import BackgroundRunner  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "fixpic"


@pytest.fixture(scope="module")
def app():
    return create_app([])


def wait_until(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    QApplication.processEvents()
    return bool(cond())


def rnd(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def fake_change(session, color=(1, 2, 3)):
    """Commit סינתטי: הופך את `session.modified` ל-True (העריכה האמיתית היא M3.4)."""
    h, w = session.rgb.shape[:2]
    em = np.zeros((h, w), np.uint8)
    em[5:25, 5:25] = 1
    out = session.rgb.copy()
    out[em > 0] = color
    session.commit(InpaintResult(out, em, RoiPlan(0, 0, w, h, 1.0, 256, w, h)))


# ------------------------------------------------------------------------------------------- strings
class TestStrings:
    def test_lookup_format_and_missing(self):
        assert strings.t("status.size", w=3, h=4) == "3×4 פיקסלים"
        assert "{name}" not in strings.t("status.saved", name="x.png")
        with pytest.raises(KeyError):
            strings.t("no.such.key")

    def test_every_key_used_in_code_exists_and_none_unused(self):
        keys = strings.all_keys()
        used: set[str] = set()
        for f in (SRC / "ui").glob("*.py"):
            used |= set(re.findall(r'"((?:menu|toolbar|action|hint|status|dialog|filter|msg|error|title|app)\.[a-z_]+)"',
                                   f.read_text(encoding="utf-8")))
        assert used - keys == set(), "מפתחות בשימוש שחסרים ב-he.json"
        assert keys - used == set(), "מחרוזות מתות ב-he.json"

    def test_all_values_nonempty_and_formattable(self):
        for k in strings.all_keys():
            v = strings.t(k)
            assert v.strip()
            names = set(re.findall(r"{(\w+)}", v))
            strings.t(k, **{n: "x" for n in names}) if names else None


# ------------------------------------------------------------------------------------------- Canvas
class TestCanvas:
    def make(self, app, w=800, h=600):
        c = Canvas()
        c.resize(w, h)
        c.show()
        QApplication.processEvents()
        return c

    def test_empty_paints_without_crash(self, app):
        c = self.make(app)
        assert not c.has_image and not c.grab().isNull()
        c.zoom_at(QPointF(10, 10), 2)
        c.fit()
        c.pan_by(QPointF(5, 5))

    def test_fit_large_and_small(self, app):
        c = self.make(app, 800, 600)
        c.set_image(rnd(4000, 6000))
        assert c.scale == pytest.approx(800 / 6000)
        iw, ih = 6000 * c.scale, 4000 * c.scale
        assert c.offset.x() == pytest.approx((800 - iw) / 2) and c.offset.y() == pytest.approx((600 - ih) / 2)
        c.set_image(rnd(80, 100))
        assert c.scale == 1.0                          # תמונה קטנה לא מוגדלת בהתאמה

    def test_zoom_keeps_point_under_cursor(self, app):
        c = self.make(app)
        c.set_image(rnd(1000, 1500))
        pos = QPointF(300, 200)
        before = c.view_to_image(pos)
        c.zoom_at(pos, 3.0)
        after = c.view_to_image(pos)
        assert after.x() == pytest.approx(before.x(), abs=1e-6) and after.y() == pytest.approx(before.y(), abs=1e-6)

    def test_zoom_limits_and_roundtrip(self, app):
        c = self.make(app)
        c.set_image(rnd(500, 500))
        for _ in range(50):
            c.zoom_at(QPointF(400, 300), 2)
        assert c.scale == MAX_SCALE
        for _ in range(100):
            c.zoom_at(QPointF(400, 300), 0.5)
        assert c.scale == MIN_SCALE
        p = QPointF(123.5, 77.25)
        q = c.view_to_image(c.image_to_view(p))
        assert q.x() == pytest.approx(p.x()) and q.y() == pytest.approx(p.y())

    def test_pan_cannot_lose_the_image(self, app):
        c = self.make(app)
        c.set_image(rnd(600, 800))
        c.pan_by(QPointF(1e6, 1e6))
        vis = c.visible_source_rect()
        assert not vis.isEmpty()
        c.pan_by(QPointF(-1e7, -1e7))
        assert not c.visible_source_rect().isEmpty()

    def test_actual_size_and_scale_signal(self, app):
        c = self.make(app)
        got: list[float] = []
        c.scaleChanged.connect(got.append)
        c.set_image(rnd(2000, 3000))
        c.actual_size()
        assert c.scale == pytest.approx(1.0) and got[-1] == pytest.approx(1.0)

    def test_pyramid_levels(self, app):
        a = rnd(4000, 6000)
        t0 = time.perf_counter()
        pyr = Pyramid(a)
        assert time.perf_counter() - t0 < 5.0
        assert np.shares_memory(pyr.arrays[0], a)                       # רמה 0 ללא העתקה
        assert min(pyr.arrays[-1].shape[:2]) <= 512
        assert pyr.level_for(2.0) == 0 and pyr.level_for(1.0) == 0
        assert pyr.level_for(0.5) == 1 and pyr.level_for(0.3) == 1 and pyr.level_for(0.1) == 3
        assert pyr.level_for(1e-6) == len(pyr.levels) - 1

    def test_renders_actual_pixels_and_alpha_background(self, app):
        c = self.make(app, 400, 300)
        img = np.zeros((100, 100, 3), np.uint8)
        img[...] = (255, 0, 0)
        c.set_image(img)
        px = c.grab().toImage().pixelColor(200, 150)
        assert (px.red(), px.green(), px.blue()) == (255, 0, 0)
        c.set_image(img, np.zeros((100, 100), np.uint8))               # שקוף לגמרי: רקע התצוגה
        px = c.grab().toImage().pixelColor(200, 150)
        assert (px.red(), px.green(), px.blue()) == (204, 204, 204)

    def test_set_image_keep_view(self, app):
        c = self.make(app)
        c.set_image(rnd(1000, 1000))
        c.zoom_at(QPointF(100, 100), 4)
        s, o = c.scale, QPointF(c.offset)
        c.set_image(rnd(1000, 1000, seed=2), keep_view=True)
        assert c.scale == s and c.offset == o
        c.set_image(rnd(500, 500), keep_view=True)                     # גודל אחר: מתאים מחדש
        assert c.scale != s

    def test_resize_refits_only_in_fit_mode(self, app):
        c = self.make(app, 800, 600)
        c.set_image(rnd(2000, 3000))
        c.resize(400, 300)
        QApplication.processEvents()
        assert c.scale == pytest.approx(400 / 3000)
        c.zoom_at(QPointF(100, 100), 2)
        s = c.scale
        c.resize(600, 500)
        QApplication.processEvents()
        assert c.scale == s

    def test_drag_pans_with_left_then_only_middle_when_disabled(self, app):
        c = self.make(app, 800, 600)
        c.set_image(rnd(1000, 1500))
        c.actual_size()
        o0 = QPointF(c.offset)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(340, 330))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(340, 330))
        assert c.offset.x() == pytest.approx(o0.x() + 40) and c.offset.y() == pytest.approx(o0.y() + 30)
        c.left_drag_pans = False
        o1 = QPointF(c.offset)
        QTest.mousePress(c, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(350, 350))
        QTest.mouseRelease(c, Qt.LeftButton, Qt.NoModifier, QPoint(350, 350))
        assert c.offset == o1
        QTest.mousePress(c, Qt.MiddleButton, Qt.NoModifier, QPoint(300, 300))
        QTest.mouseMove(c, QPoint(320, 310))
        QTest.mouseRelease(c, Qt.MiddleButton, Qt.NoModifier, QPoint(320, 310))
        assert c.offset.x() == pytest.approx(o1.x() + 20) and c.offset.y() == pytest.approx(o1.y() + 10)

    def test_wheel_zooms_at_cursor(self, app):
        c = self.make(app)
        c.set_image(rnd(1000, 1500))
        pos = QPointF(250, 180)
        before = c.view_to_image(pos)
        s0 = c.scale
        ev = QWheelEvent(pos, c.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                         Qt.NoScrollPhase, False)
        c.wheelEvent(ev)
        assert c.scale == pytest.approx(s0 * 1.25)
        after = c.view_to_image(pos)
        assert after.x() == pytest.approx(before.x(), abs=1e-6)

    def test_24mp_paint_is_fast_enough_to_be_interactive(self, app):
        c = self.make(app, 1600, 900)
        c.set_image(np.zeros((4000, 6000, 3), np.uint8))
        times = []
        for i in range(10):
            c.pan_by(QPointF(30, 10))
            t0 = time.perf_counter()
            c.grab()
            times.append(time.perf_counter() - t0)
        assert np.median(times) < 0.25            # רינדור תוכנה offscreen, גבול רופף; אין משמעות לחומרת היעד


# ------------------------------------------------------------------------------------------- convert / workers
def test_qimage_to_array_values_and_row_padding(app):
    q = QImage(5, 3, QImage.Format_RGB888)         # 5*3 = 15 בייט לשורה -> ריפוד ל-16
    q.fill(0)
    q.setPixelColor(4, 2, __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(10, 20, 30))
    a = qimage_to_array(q)
    assert a.shape == (3, 5, 4) and a.dtype == np.uint8 and a.flags.owndata
    assert tuple(a[2, 4]) == (10, 20, 30, 255) and tuple(a[0, 0]) == (0, 0, 0, 255)
    with pytest.raises(InvalidInputError):
        qimage_to_array(QImage())


def test_runner_delivers_on_ui_thread(app):
    r = BackgroundRunner()
    got: dict = {}
    main = threading.current_thread()

    def work():
        got["worker"] = threading.current_thread()
        return 42

    def done(v):
        got["done_thread"], got["v"] = threading.current_thread(), v
    r.submit(work, done, lambda e: got.setdefault("err", e))
    assert wait_until(lambda: "v" in got)
    assert got["v"] == 42 and got["worker"] is not main and got["done_thread"] is main
    r.submit(lambda: 1 / 0, lambda v: None, lambda e: got.__setitem__("err", e))
    assert wait_until(lambda: "err" in got) and isinstance(got["err"], ZeroDivisionError)
    assert r.active == 0


# ------------------------------------------------------------------------------------------- MainWindow
class Win:
    """עוטף MainWindow עם דיאלוגים מוחלפים שרושמים קריאות."""

    def __init__(self):
        self.w = MainWindow()
        self.errors: list[tuple[str, dict]] = []
        self.discard_answer = "discard"
        self.asked = 0
        self.open_pick = ""
        self.save_pick: tuple[str, str] = ("", "")
        self.save_picks = 0
        w = self.w
        w._error = lambda key, **f: self.errors.append((key, f))
        w._ask_discard = self._ask
        w._pick_open = lambda: self.open_pick
        w._pick_save = self._pick_save
        w.show()

    def _ask(self):
        self.asked += 1
        return self.discard_answer

    def _pick_save(self, suggested, selected):
        self.save_picks += 1
        self.last_suggested, self.last_filter = suggested, selected
        return self.save_pick

    def idle(self, timeout=10.0):
        return wait_until(lambda: not self.w.busy, timeout)

    def open(self, path):
        self.w.open_path(path)
        assert self.idle()


@pytest.fixture()
def win(app):
    h = Win()
    yield h
    h.w._runner.wait_idle()
    h.w.session = None
    h.w.close()


def png(tmp_path, name="תמונה.png", h=120, w=160, seed=0, alpha=None):
    d = tmp_path / "תיקייה שלי"
    d.mkdir(exist_ok=True)
    p = d / name
    save_image(p, rnd(h, w, seed), alpha)
    return p


class TestMainWindow:
    def test_rtl_and_initial_state(self, app, win):
        assert QApplication.layoutDirection() == Qt.RightToLeft and win.w.layoutDirection() == Qt.RightToLeft
        assert win.w.session is None and not win.w.act_save.isEnabled() and win.w.act_open.isEnabled()
        assert win.w.windowTitle() == strings.t("app.name")
        assert not win.w.act_fit.isEnabled()

    def test_open_hebrew_path_background(self, win, tmp_path):
        p = png(tmp_path)
        win.w.open_path(p)
        assert win.w.busy and win.w.progress.isVisible() or win.w.busy        # עסוק בזמן הטעינה (לא נחסם ה-UI)
        assert win.idle()
        s = win.w.session
        assert s is not None and np.array_equal(s.rgb, load_image(p).rgb)
        assert win.w.canvas.has_image and p.name in win.w.windowTitle()
        assert "160×120" in win.w.lbl_size.text() and "%" in win.w.lbl_zoom.text()
        assert win.w.act_save.isEnabled() and not win.w.progress.isVisible() and win.errors == []

    def test_open_via_dialog(self, win, tmp_path):
        win.open_pick = str(png(tmp_path))
        win.w.open_dialog()
        assert win.idle() and win.w.session is not None
        win.open_pick = ""
        win.w.open_dialog()                                  # ביטול הדיאלוג
        assert not win.w.busy

    def test_open_bad_file_shows_error_and_keeps_current(self, win, tmp_path):
        good = png(tmp_path)
        win.open(good)
        keep = win.w.session
        bad = tmp_path / "שבור.png"
        bad.write_bytes(b"not an image")
        win.open(bad)
        assert [k for k, _ in win.errors] == ["error.load"] and win.w.session is keep
        win.open(tmp_path / "אין-כזה.png")
        assert [k for k, _ in win.errors][-1] == "error.load" and win.w.session is keep and not win.w.busy

    def test_open_while_busy_is_refused(self, win, tmp_path):
        p = png(tmp_path)
        win.w.busy = True
        win.w.open_path(p)
        assert [k for k, _ in win.errors] == ["msg.busy"] and win.w.session is None
        win.w.busy = False

    def test_drop_local_file_and_non_local(self, win, tmp_path):
        p = png(tmp_path)
        md = QMimeData()
        md.setUrls([QUrl.fromLocalFile(str(p))])
        ev = QDropEvent(QPointF(10, 10), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
        win.w.dropEvent(ev)
        assert win.idle() and win.w.session is not None
        md2 = QMimeData()
        md2.setUrls([QUrl("https://example.com/a.png")])
        win.w.dropEvent(QDropEvent(QPointF(10, 10), Qt.CopyAction, md2, Qt.LeftButton, Qt.NoModifier))
        assert win.errors[-1][0] == "error.not_image_drop"

    def test_paste_image_and_empty_clipboard(self, win):
        cb = QGuiApplication.clipboard()
        q = QImage(60, 40, QImage.Format_RGB888)
        q.fill(Qt.green)
        cb.setImage(q)
        win.w.paste()
        s = win.w.session
        assert s is not None and s.size == (60, 40) and (s.rgb[10, 10] == (0, 255, 0)).all() and s.alpha is None
        assert win.w.windowTitle().startswith(strings.t("title.untitled"))
        cb.clear()
        win.w.session = None
        win.w.paste()
        assert win.errors[-1][0] == "error.clipboard_empty"

    def test_paste_with_transparency_keeps_alpha(self, win):
        q = QImage(30, 20, QImage.Format_ARGB32)
        q.fill(Qt.transparent)
        QGuiApplication.clipboard().setImage(q)
        win.w.paste()
        assert win.w.session.alpha is not None and (win.w.session.alpha == 0).all()

    def test_save_as_writes_file_adds_extension_and_marks_saved(self, win, tmp_path):
        win.open(png(tmp_path))
        s = win.w.session
        fake_change(s)
        win.w._refresh()                                     # ב-M3.4 ה-UI יקרא לזה אחרי כל Commit
        assert s.modified and "*" in win.w.windowTitle()
        out = tmp_path / "תיקייה שלי" / "פלט"
        win.save_pick = (str(out), strings.t("filter.webp"))
        win.w.save_as()
        assert win.w.busy and win.idle()
        res = out.with_suffix(".webp")
        assert res.exists() and not s.modified and win.w.save_path == res and win.errors == []
        assert "*" not in win.w.windowTitle() and res.name in win.w.windowTitle()
        assert load_image(res).size == (160, 120)

    def test_save_suggests_name_and_source_format(self, win, tmp_path):
        d = tmp_path / "תיקייה שלי"
        d.mkdir()
        j = d / "צילום.jpg"
        save_image(j, rnd(), fmt="JPEG")
        win.open(j)
        win.save_pick = ("", "")
        win.w.save_as()
        assert win.last_suggested.endswith("צילום.jpg") and win.last_filter == strings.t("filter.jpg")

    def test_ctrl_s_never_overwrites_original_then_reuses_path(self, win, tmp_path):
        src = png(tmp_path)
        before = src.read_bytes()
        win.open(src)
        fake_change(win.w.session)
        out = tmp_path / "תיקייה שלי" / "חדש.png"
        win.save_pick = (str(out), strings.t("filter.png"))
        win.w.save()                                         # אין נתיב קודם: נפתח "שמירה בשם"
        assert win.save_picks == 1 and win.idle()
        assert src.read_bytes() == before and out.exists()
        fake_change(win.w.session, (9, 9, 9))
        win.w.save()                                         # הפעם בלי דיאלוג
        assert win.save_picks == 1 and win.idle() and not win.w.session.modified
        assert (load_image(out).rgb[10, 10] == (9, 9, 9)).all()

    def test_save_cancel_dialog_does_nothing(self, win, tmp_path):
        win.open(png(tmp_path))
        fake_change(win.w.session)
        win.save_pick = ("", "")
        win.w.save()
        assert not win.w.busy and win.w.session.modified

    def test_save_failure_keeps_modified_and_reports(self, win, tmp_path):
        win.open(png(tmp_path))
        fake_change(win.w.session)
        win.save_pick = (str(tmp_path / "אין תיקייה" / "x.png"), strings.t("filter.png"))
        win.w.save_as()
        assert win.idle()
        assert win.errors[-1][0] == "error.save" and win.w.session.modified and win.w.save_path is None

    def test_save_jpeg_of_transparent_reports_flatten(self, win, tmp_path):
        a = np.zeros((120, 160), np.uint8)
        a[:, :80] = 255
        win.open(png(tmp_path, alpha=a))
        out = tmp_path / "תיקייה שלי" / "שטוח.jpg"
        win.save_pick = (str(out), strings.t("filter.jpg"))
        win.w.save_as()
        assert win.idle() and out.exists()
        assert win.w.statusBar().currentMessage() == strings.t("msg.alpha_flattened")

    def test_change_during_save_stays_modified(self, win, tmp_path):
        win.open(png(tmp_path))
        s = win.w.session
        fake_change(s)
        out = tmp_path / "תיקייה שלי" / "x.png"
        win.w._start_save(out, None)
        fake_change(s, (7, 7, 7))                            # שינוי בזמן שהשמירה רצה
        assert win.idle() and s.modified                     # נשמר מצב קודם, ולכן עדיין "שונה"
        assert (load_image(out).rgb[10, 10] == (1, 2, 3)).all()

    def test_discard_prompt_on_open(self, win, tmp_path):
        a, b = png(tmp_path, "א.png", seed=1), png(tmp_path, "ב.png", seed=2)
        win.open(a)
        fake_change(win.w.session)
        keep = win.w.session
        win.discard_answer = "cancel"
        win.open(b)
        assert win.asked == 1 and win.w.session is keep
        win.discard_answer = "discard"
        win.open(b)
        assert win.asked == 2 and win.w.session is not keep and win.w.display_name == "ב.png"

    def test_discard_prompt_save_then_open(self, win, tmp_path):
        a, b = png(tmp_path, "א.png", seed=1), png(tmp_path, "ב.png", seed=2)
        win.open(a)
        fake_change(win.w.session)
        win.discard_answer = "save"
        out = tmp_path / "תיקייה שלי" / "נשמר.png"
        win.save_pick = (str(out), strings.t("filter.png"))
        win.w.open_path(b)
        assert wait_until(lambda: win.w.display_name == "ב.png" and not win.w.busy)
        assert out.exists() and win.errors == []

    def test_no_prompt_when_unmodified(self, win, tmp_path):
        win.open(png(tmp_path, "א.png"))
        win.open(png(tmp_path, "ב.png", seed=3))
        assert win.asked == 0

    def test_close_event_rules(self, win, tmp_path):
        from PySide6.QtGui import QCloseEvent
        win.open(png(tmp_path))
        ev = QCloseEvent()
        win.w.closeEvent(ev)
        assert ev.isAccepted()                               # לא שונה
        fake_change(win.w.session)
        win.discard_answer = "cancel"
        ev = QCloseEvent()
        win.w.closeEvent(ev)
        assert not ev.isAccepted() and win.asked == 1
        win.w.busy = True
        ev = QCloseEvent()
        win.w.closeEvent(ev)
        assert not ev.isAccepted() and win.errors[-1][0] == "msg.busy"
        win.w.busy = False

    def test_zoom_actions_update_status(self, win, tmp_path):
        win.open(png(tmp_path, h=1000, w=1500))
        z0 = win.w.canvas.scale
        win.w.act_zoom_in.trigger()
        assert win.w.canvas.scale == pytest.approx(z0 * 1.25)
        win.w.act_actual.trigger()
        assert win.w.lbl_zoom.text() == strings.t("status.zoom", percent=100)
        win.w.act_fit.trigger()
        assert win.w.canvas.scale == pytest.approx(z0)

    def test_large_image_open_and_save_smoke(self, win, tmp_path):
        p = tmp_path / "גדולה.png"
        save_image(p, np.random.default_rng(0).integers(0, 256, (2000, 3000, 3), dtype=np.uint8))
        win.open(p)
        assert win.w.session.size == (3000, 2000)
        out = tmp_path / "גדולה-פלט.jpg"
        win.save_pick = (str(out), strings.t("filter.jpg"))
        win.w.save_as()
        assert win.idle(60) and out.exists()
