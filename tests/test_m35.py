"""M3.5: Undo/Redo והשוואה לפני/אחרי בממשק (PARITY-SPEC §2 ו-§3). Qt offscreen, Scheduler אמיתי, מודל מדומה.

מוכיח את החיבור לממשק (פעולות, מקשים, רענון קנבס, כותרת, זום נשמר, 20 צעדים, ביטול בזמן עיבוד) ואת ההשוואה בפיקסלים.
לא נבדק: Windows, מקלדת עברית (קיצורי `C`/`B` נקבעים לפי האות, ובפריסה עברית עשויים לא לפעול; Ctrl+Z סטנדרטי),
מראה בעין, i5-6500T."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fixpic.imgio import image_from_array  # noqa: E402
from fixpic.session import HistoryAction  # noqa: E402
from test_engine import FILL, FakeModel, rnd  # noqa: E402
from test_m34 import (EWin, app, drag, grab_rgb, scr, wait_until, ew, ew_gate, ew_noengine, T,  # noqa: E402,F401
                      _close_canvases)  # noqa: F401
from test_scheduler import GateModel  # noqa: E402

pytestmark = pytest.mark.usefixtures("_close_canvases")


def view_1to1(canvas):
    """זום 100% והתחלה ב-(0, 0), כדי שפיקסל מסך = פיקסל תמונה."""
    canvas.scale, canvas.offset, canvas._fit_mode = 1.0, QPointF(0, 0), False
    canvas.update()
    QApplication.processEvents()


def shot_region(canvas, h, w, keep_hover=False):
    """צילום אזור התמונה. עיגול המברשת (שעוקב אחרי העכבר במצב ציור) מוסר אלא אם התבקש, כדי להשוות פיקסלים נקיים."""
    if not keep_hover:
        canvas._hover = None
    return grab_rgb(canvas)[:h, :w]


def commit_stroke(ew, x, y=50):
    drag(ew.canvas, [(x, y), (x + 10, y)])
    assert ew.idle()


# ---------------------------------------------------------------------------- Undo / Redo
class TestUndoRedo:
    def test_undo_redo_actions_walk_the_history_and_refresh_canvas_title_and_buttons(self, ew):
        s = ew.install(h=200, w=400)
        w = ew.w
        assert not w.act_undo.isEnabled() and not w.act_redo.isEnabled()
        snaps = [s.rgb.copy()]
        for x in (50, 150, 250):
            commit_stroke(ew, x, 100)
            snaps.append(s.rgb.copy())
        assert w.act_undo.isEnabled() and not w.act_redo.isEnabled() and "*" in w.windowTitle()
        for i in (2, 1, 0):
            w.act_undo.trigger()
            assert np.array_equal(s.rgb, snaps[i])
            assert np.array_equal(ew.canvas.pyr.arrays[0], snaps[i])             # הקנבס מציג את המצב המבוטל
            assert w.act_redo.isEnabled()
        assert not w.act_undo.isEnabled() and "*" not in w.windowTitle()        # חזרה למצב הנטען = לא שונה
        for i in (1, 2, 3):
            w.act_redo.trigger()
            assert np.array_equal(s.rgb, snaps[i])
            assert np.array_equal(ew.canvas.pyr.arrays[0], snaps[i])
        assert not w.act_redo.isEnabled() and w.act_undo.isEnabled() and "*" in w.windowTitle()

    def test_twenty_steps_undo_and_redo_exactly(self, ew):
        """DoD של M3: 20 צעדי Undo/Redo, כל מצב ביניים זהה לבייט למה שנרשם ב-Commit."""
        s = ew.install(h=100, w=1300, diameter=16)
        snaps = [s.rgb.copy()]
        for i in range(20):
            commit_stroke(ew, 40 + 60 * i, 50)
            snaps.append(s.rgb.copy())
        assert len({a.tobytes() for a in snaps}) == 21                          # כל צעד שינה את התמונה
        for i in range(19, -1, -1):
            ew.w.act_undo.trigger()
            assert np.array_equal(s.rgb, snaps[i]), f"Undo ל-{i}"
        assert not ew.w.act_undo.isEnabled()
        for i in range(1, 21):
            ew.w.act_redo.trigger()
            assert np.array_equal(s.rgb, snaps[i]), f"Redo ל-{i}"
        assert not ew.w.act_redo.isEnabled()

    def test_keyboard_shortcuts(self, ew):
        s = ew.install(h=200, w=400)
        before = s.rgb.copy()
        commit_stroke(ew, 100, 100)
        after = s.rgb.copy()
        ew.w.activateWindow()
        ew.canvas.setFocus()
        QApplication.processEvents()
        QTest.keyClick(ew.canvas, Qt.Key_Z, Qt.ControlModifier)
        assert np.array_equal(s.rgb, before)
        QTest.keyClick(ew.canvas, Qt.Key_Y, Qt.ControlModifier)
        assert np.array_equal(s.rgb, after)
        QTest.keyClick(ew.canvas, Qt.Key_Z, Qt.ControlModifier)
        QTest.keyClick(ew.canvas, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
        assert np.array_equal(s.rgb, after)                                      # Ctrl+Shift+Z = Redo

    def test_new_stroke_after_undo_drops_redo(self, ew):
        ew.install(h=200, w=400)
        commit_stroke(ew, 80, 100)
        ew.w.act_undo.trigger()
        assert ew.w.act_redo.isEnabled()
        commit_stroke(ew, 250, 100)
        assert not ew.w.act_redo.isEnabled()

    def test_view_survives_undo_redo(self, ew):
        ew.install(h=200, w=400)
        commit_stroke(ew, 100, 100)
        ew.canvas.zoom_at(QPointF(200, 150), 2.0)
        sc, off = ew.canvas.scale, QPointF(ew.canvas.offset)
        ew.w.act_undo.trigger()
        assert ew.canvas.scale == sc and ew.canvas.offset == off
        ew.w.act_redo.trigger()
        assert ew.canvas.scale == sc and ew.canvas.offset == off

    def test_undo_during_processing_cancels_the_job_and_keeps_earlier_commits(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=200, w=400)
        drag(ew.canvas, [(60, 100)])
        assert ew.model.entered.acquire(timeout=T)
        ew.model.gate.release(1)
        assert ew.idle()
        done = s.rgb.copy()
        drag(ew.canvas, [(250, 100)])
        assert ew.model.entered.acquire(timeout=T)
        assert ew.w.act_undo.isEnabled()
        ew.w.act_undo.trigger()                                                  # בזמן עיבוד: מבטל את העבודה
        assert not s.busy and s.mask.empty
        ew.model.gate.release(5)
        assert wait_until(lambda: ew.sched.current is None)
        QApplication.processEvents()
        assert np.array_equal(s.rgb, done) and s.history.can_undo               # ה-Commit הראשון לא בוטל
        ew.w.act_undo.trigger()                                                  # עכשיו כן: מבטל את הראשון
        assert not s.history.can_undo and not s.modified

    def test_redo_is_rejected_while_processing(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=200, w=400)
        drag(ew.canvas, [(60, 100)])
        assert ew.model.entered.acquire(timeout=T)
        ew.model.gate.release(1)
        assert ew.idle()
        ew.w.act_undo.trigger()
        assert s.history.can_redo and ew.w.act_redo.isEnabled()
        drag(ew.canvas, [(250, 100)])                                            # עבודה רצה; ה-Redo עדיין ברשימה
        assert ew.model.entered.acquire(timeout=T) and s.busy and s.history.can_redo
        assert not ew.w.act_redo.isEnabled()
        before = s.rgb.copy()
        assert ew.c.redo() is HistoryAction.BUSY
        assert "בזמן עיבוד" in ew.w.statusBar().currentMessage() and np.array_equal(s.rgb, before)
        ew.model.gate.release(5)
        assert ew.idle()

    def test_undo_drops_a_mask_waiting_for_the_queue(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=300, w=700)
        for x in (40, 130, 220, 310):
            drag(ew.canvas, [(x, 100)])
            if x == 40:
                assert ew.model.entered.acquire(timeout=T)
        drag(ew.canvas, [(400, 100)])
        assert ew.c.waiting == "queue"
        ew.w.act_undo.trigger()
        assert ew.c.waiting is None and s.mask.empty and not s.busy

    def test_undo_is_ignored_in_the_middle_of_a_stroke(self, ew):
        s = ew.install(h=200, w=400)
        commit_stroke(ew, 80, 100)
        done = s.rgb.copy()
        QTest.mousePress(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 250, 100))
        assert ew.c.stroking and not ew.w.act_undo.isEnabled()
        assert ew.c.undo() is HistoryAction.NOTHING
        QTest.mouseRelease(ew.canvas, Qt.LeftButton, Qt.NoModifier, scr(ew.canvas, 250, 100))
        assert ew.idle() and not np.array_equal(s.rgb, done)                     # המשיכה הסתיימה כרגיל

    def test_nothing_to_undo_or_redo_is_a_noop(self, ew):
        s = ew.install()
        v = s.version
        assert ew.c.undo() is HistoryAction.NOTHING and ew.c.redo() is HistoryAction.NOTHING
        assert s.version == v and not s.modified

    def test_no_session_actions_disabled(self, ew_noengine):
        w = ew_noengine.w
        assert not w.act_undo.isEnabled() and not w.act_redo.isEnabled() and not w.act_compare.isEnabled()
        w._undo()
        w._redo()                                                                # בלי קריסה

    def test_undo_while_loading_or_saving_is_blocked(self, ew):
        s = ew.install(h=200, w=400)
        commit_stroke(ew, 80, 100)
        done = s.rgb.copy()
        ew.w._set_busy(True)
        assert not ew.w.act_undo.isEnabled()
        ew.w._undo()
        assert np.array_equal(s.rgb, done)
        ew.w._set_busy(False)
        assert ew.w.act_undo.isEnabled()

    def test_undo_message_and_mask_panel_history_are_independent(self, ew):
        s = ew.install(h=200, w=400)
        commit_stroke(ew, 80, 100)
        ew.w.act_undo.trigger()
        assert "בוטלה" in ew.w.statusBar().currentMessage()
        ew.w.act_redo.trigger()
        assert "שוחזרה" in ew.w.statusBar().currentMessage()
        assert s.mask.empty                                                      # Undo של תמונה לא נוגע במסכה


# ---------------------------------------------------------------------------- השוואה לפני/אחרי
def edited_window(ew, h=300, w=300):
    s = ew.install(h=h, w=w, seed=5, diameter=40)
    view_1to1(ew.canvas)
    drag(ew.canvas, [(100, 150), (200, 150)])
    assert ew.idle()
    view_1to1(ew.canvas)
    return s


class TestCompare:
    def test_toggle_shows_original_then_edited_exactly(self, ew):
        s = edited_window(ew)
        orig, edited = s.original_rgb, s.rgb
        assert not np.array_equal(orig, edited)
        assert np.array_equal(shot_region(ew.canvas, 300, 300), edited)
        ew.w.act_compare.trigger()
        QApplication.processEvents()
        assert ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), orig)
        assert not ew.w.lbl_cmp.isHidden() and "מקורית" in ew.w.lbl_cmp.text()
        ew.w.act_compare.trigger()
        QApplication.processEvents()
        assert not ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), edited)
        assert ew.w.lbl_cmp.isHidden()

    def test_hold_with_key_B_and_release(self, ew):
        s = edited_window(ew)
        ew.canvas.setFocus()
        QTest.keyPress(ew.canvas, Qt.Key_B)
        assert ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.original_rgb)
        QTest.keyRelease(ew.canvas, Qt.Key_B)
        assert not ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.rgb)

    def test_hold_with_toolbar_button_press_and_release(self, ew):
        s = edited_window(ew)
        QTest.mousePress(ew.w.btn_hold, Qt.LeftButton)
        assert ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.original_rgb)
        QTest.mouseRelease(ew.w.btn_hold, Qt.LeftButton)
        assert not ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.rgb)

    def test_hold_is_released_when_focus_is_lost(self, ew):
        edited_window(ew)
        ew.canvas.setFocus()
        QTest.keyPress(ew.canvas, Qt.Key_B)
        assert ew.canvas.comparing
        ew.w.brush_panel.spin.setFocus()                                         # הפוקוס עבר: שחרור המקש לא יגיע
        QApplication.processEvents()
        assert not ew.canvas.comparing

    def test_toggle_and_hold_combine(self, ew):
        edited_window(ew)
        ew.w.act_compare.trigger()
        ew.canvas.setFocus()
        QTest.keyPress(ew.canvas, Qt.Key_B)
        QTest.keyRelease(ew.canvas, Qt.Key_B)
        assert ew.canvas.comparing                                               # המתג עדיין דלוק
        ew.w.act_compare.trigger()
        assert not ew.canvas.comparing

    def test_no_painting_while_comparing(self, ew):
        s = edited_window(ew)
        ew.w.act_compare.trigger()
        assert ew.canvas.painting is False
        starts = []
        ew.canvas.strokeStarted.connect(lambda p, e: starts.append(1))
        before = s.rgb.copy()
        drag(ew.canvas, [(40, 40), (60, 40)])
        assert starts == [] and s.mask.empty and np.array_equal(s.rgb, before)
        ew.w.act_compare.trigger()
        assert ew.canvas.painting is True
        drag(ew.canvas, [(40, 40), (60, 40)])
        assert starts == [1]

    def test_mask_overlay_and_brush_ring_hidden_while_comparing(self, ew_noengine):
        ew = ew_noengine
        s = ew.install(h=300, w=300, seed=5, diameter=40)
        view_1to1(ew.canvas)
        drag(ew.canvas, [(100, 150), (200, 150)])                                # בלי מנוע: המסכה נשארת מצוירת
        assert not s.mask.empty
        QTest.mouseMove(ew.canvas, QPoint(150, 150))
        view_1to1(ew.canvas)
        assert not np.array_equal(shot_region(ew.canvas, 300, 300), s.rgb)       # בפועל: אדום מעל התמונה
        ring = shot_region(ew.canvas, 300, 300, keep_hover=True)
        ew.w.act_compare.trigger()
        QTest.mouseMove(ew.canvas, QPoint(151, 151))
        assert np.array_equal(shot_region(ew.canvas, 300, 300, keep_hover=True), s.original_rgb)  # נקי: בלי מסכה ובלי עיגול
        assert not np.array_equal(ring, s.original_rgb)

    def test_badge_is_drawn_only_while_comparing(self, ew):
        edited_window(ew)
        ew.w.act_compare.trigger()
        with_badge = grab_rgb(ew.canvas)
        ew.canvas.set_compare(True, ew.s.original_rgb, ew.s.original_alpha, "")
        without = grab_rgb(ew.canvas)
        assert not np.array_equal(with_badge, without)
        diff = np.nonzero((with_badge != without).any(axis=2))
        assert diff[0].max() < 80 and diff[1].min() > ew.canvas.width() // 2     # בפינה העליונה הימנית בלבד

    def test_original_pyramid_built_once_and_reset_for_a_new_image(self, ew):
        edited_window(ew)
        ew.w.act_compare.trigger()
        first = ew.canvas._cmp_pyr
        assert first is not None
        ew.w.act_compare.trigger()
        ew.w.act_compare.trigger()
        assert ew.canvas._cmp_pyr is first                                       # לא נבנתה מחדש
        ew.install(h=200, w=200, seed=9)
        assert ew.canvas._cmp_pyr is None and not ew.canvas.comparing and not ew.w.act_compare.isChecked()
        assert ew.w.lbl_cmp.isHidden()

    def test_compare_with_no_edits_is_identical(self, ew):
        s = ew.install(h=300, w=300, seed=2)
        view_1to1(ew.canvas)
        a = shot_region(ew.canvas, 300, 300)
        ew.w.act_compare.trigger()
        assert np.array_equal(shot_region(ew.canvas, 300, 300), a) and np.array_equal(a, s.rgb)

    def test_compare_keeps_zoom_and_pan(self, ew):
        edited_window(ew)
        ew.canvas.zoom_at(QPointF(150, 150), 2.0)
        sc, off = ew.canvas.scale, QPointF(ew.canvas.offset)
        ew.w.act_compare.trigger()
        assert ew.canvas.scale == sc and ew.canvas.offset == off
        ew.w.act_compare.trigger()
        assert ew.canvas.scale == sc and ew.canvas.offset == off

    def test_undo_and_commit_while_comparing_keep_showing_the_original(self, ew):
        s = edited_window(ew)
        ew.w.act_compare.trigger()
        ew.w.act_undo.trigger()                                                  # שינוי העובדת בזמן השוואה
        view_1to1(ew.canvas)
        assert ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.original_rgb)
        ew.w.act_compare.trigger()
        view_1to1(ew.canvas)
        assert np.array_equal(shot_region(ew.canvas, 300, 300), s.rgb) and np.array_equal(s.rgb, s.original_rgb)
        ew.w.act_redo.trigger()
        view_1to1(ew.canvas)
        assert not np.array_equal(shot_region(ew.canvas, 300, 300), s.original_rgb)

    def test_job_finishing_while_comparing_updates_the_working_image_underneath(self, ew_gate):
        ew = ew_gate
        s = ew.install(h=300, w=300, seed=5, diameter=40)
        view_1to1(ew.canvas)
        drag(ew.canvas, [(100, 150), (200, 150)])
        assert ew.model.entered.acquire(timeout=T)
        ew.w.act_compare.trigger()                                               # מסתכלים על המקור בזמן העיבוד
        ew.model.gate.release(1)
        assert wait_until(lambda: s.modified and not s.busy)
        view_1to1(ew.canvas)
        assert ew.canvas.comparing and np.array_equal(shot_region(ew.canvas, 300, 300), s.original_rgb)
        ew.w.act_compare.trigger()
        view_1to1(ew.canvas)
        assert np.array_equal(shot_region(ew.canvas, 300, 300), s.rgb)           # וכשחוזרים רואים את התוצאה

    def test_alpha_original_is_composited_like_the_working_image(self, ew_noengine):
        ew = ew_noengine
        arr = np.dstack([rnd(120, 120, 3), np.full((120, 120), 255, np.uint8)])
        arr[:40, :, 3] = 0
        ew.w._install(image_from_array(arr), "a.png")
        view_1to1(ew.canvas)
        base = shot_region(ew.canvas, 120, 120)
        ew.w.act_compare.trigger()
        assert np.array_equal(shot_region(ew.canvas, 120, 120), base)            # אותו הרכבה על רקע השקיפות
        assert tuple(base[10, 10]) == (204, 204, 204)

    def test_compare_actions_enabled_with_image_only(self, ew_noengine):
        w = ew_noengine.w
        assert not w.act_compare.isEnabled() and not w.btn_hold.isEnabled()
        ew_noengine.install()
        assert w.act_compare.isEnabled() and w.btn_hold.isEnabled()

    def test_hold_key_with_modifier_or_autorepeat_is_ignored(self, ew):
        edited_window(ew)
        ew.canvas.setFocus()
        QTest.keyPress(ew.canvas, Qt.Key_B, Qt.ControlModifier)
        assert not ew.canvas.comparing
        QTest.keyRelease(ew.canvas, Qt.Key_B, Qt.ControlModifier)
