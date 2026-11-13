"""M3.2: שירותי אפליקציה ללא UI: MaskLayer, Patch/History, Session (Commit אטומי, Undo/Redo, Alpha, Race)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fixpic.engine import JobScheduler, JobState
from fixpic.errors import (EmptyMaskError, FixPicError, InferenceError, InvalidInputError, OutOfMemoryError,
                           QueueFullError)
from fixpic.history import History, Patch
from fixpic.imgio import image_from_array, load_image
from fixpic.models import ModelInfo
from fixpic.pipeline import InpaintResult, Pipeline, PipelineConfig
from fixpic.pipeline.roi import RoiPlan
from fixpic.session import HistoryAction, MaskLayer, OutcomeKind, Session, fill_alpha
from fixpic.session import session as session_mod
from test_backends import tiny  # noqa: F401  (fixture)
from test_scheduler import GateModel

T = 5.0


def rnd(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def box_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def fake_result(base: np.ndarray, em: np.ndarray, color) -> InpaintResult:
    """תוצאת Inpaint סינתטית: צבע קבוע בתוך המסכה, בסיס מחוץ לה."""
    out = base.copy()
    out[em > 0] = color
    roi = RoiPlan(0, 0, base.shape[1], base.shape[0], 1.0, 256, base.shape[1], base.shape[0])
    return InpaintResult(out, em, roi)


def session(h=120, w=160, alpha=None, seed=0, **kw):
    return Session(image_from_array(rnd(h, w, seed)) if alpha is None else _with_alpha(h, w, alpha, seed), **kw)


def _with_alpha(h, w, alpha, seed):
    li = image_from_array(rnd(h, w, seed))
    li.alpha = alpha
    return li


# ------------------------------------------------------------------------------------------- MaskLayer
class TestMaskLayer:
    def test_dot_and_stroke(self):
        m = MaskLayer(100, 200)
        assert m.empty and m.bbox() is None and m.brush_px is None
        r = m.paint([(50, 50)], 11)
        assert r is not None and not m.empty
        assert 80 <= m.pixel_count <= 130                           # עיגול בקוטר 11
        assert m.array[50, 50] == 255 and m.array[50, 70] == 0
        m.clear()
        m.paint([(10, 10), (100, 10)], 5)
        assert m.bbox() == (7, 7, 104, 14) and m.array[10, 55] == 255

    def test_values_are_binary_and_uint8(self):
        m = MaskLayer(50, 50)
        m.paint([(5, 5), (40, 30)], 9)
        assert m.array.dtype == np.uint8 and set(np.unique(m.array)) == {0, 255}

    def test_erase(self):
        m = MaskLayer(80, 80)
        m.paint([(10, 40), (70, 40)], 11)
        m.erase([(40, 40)], 15)
        assert m.array[40, 40] == 0 and m.array[40, 15] == 255 and m.array[40, 65] == 255
        m.erase([(0, 0), (79, 79)], 200)
        assert m.empty

    def test_clipping_and_outside(self):
        m = MaskLayer(50, 60)
        assert m.paint([(-500, -500), (-400, -400)], 5) is None and m.empty and m.version == 0
        r = m.paint([(-10, 25), (30, 25)], 10)
        assert r is not None and r[0] == 0 and not m.empty        # נחתך בקצה, לא נכשל
        m.clear()
        m.paint([(55, 45), (200, 200)], 10)
        assert m.array[45, 58] == 255

    def test_returned_rect_covers_changes(self):
        m = MaskLayer(100, 100)
        x0, y0, x1, y1 = m.paint([(20, 30), (60, 70)], 13)
        ys, xs = np.nonzero(m.array)
        assert x0 <= xs.min() and xs.max() < x1 and y0 <= ys.min() and ys.max() < y1

    def test_brush_px_is_max_painted_and_ignores_erase(self):
        m = MaskLayer(100, 100)
        m.paint([(10, 10)], 8)
        m.paint([(50, 50)], 20)
        m.paint([(80, 80)], 5)
        m.erase([(50, 50)], 99)
        assert m.brush_px == 20.0
        m.clear()
        assert m.brush_px is None

    def test_take_snapshot_independent_and_array_readonly(self):
        m = MaskLayer(40, 40)
        m.paint([(20, 20)], 9)
        snap = m.snapshot()
        m.clear()
        assert snap.any() and m.empty
        m.paint([(5, 5)], 5)
        with pytest.raises(ValueError):
            m.array[0, 0] = 255
        t = m.take()
        assert t.any() and m.empty and m.brush_px is None

    def test_add_or_semantics_and_validation(self):
        m = MaskLayer(30, 30)
        m.paint([(5, 5)], 5)
        other = np.zeros((30, 30), np.uint8)
        other[20:25, 20:25] = 1                                      # כל ערך חיובי נחשב
        m.add(other)
        assert m.array[5, 5] == 255 and m.array[22, 22] == 255 and set(np.unique(m.array)) == {0, 255}
        with pytest.raises(InvalidInputError):
            m.add(np.zeros((10, 10), np.uint8))

    @pytest.mark.parametrize("pts,d", [([], 5), ([(float("nan"), 1)], 5), ([(1, float("inf"))], 5),
                                       ([(1e9, 1)], 5), ([(1, 1)], 0), ([(1, 1)], -3), ([(1, 1)], float("nan")),
                                       ([("a", 1)], 5), ([(1, 1)], "x"), ([5], 5)])
    def test_invalid_inputs(self, pts, d):
        m = MaskLayer(20, 20)
        with pytest.raises(InvalidInputError):
            m.paint(pts, d)
        assert m.empty and m.version == 0

    def test_version_and_count_cache(self):
        m = MaskLayer(30, 30)
        v = m.version
        m.paint([(10, 10)], 5)
        assert m.version == v + 1
        c1 = m.pixel_count
        m.paint([(20, 20)], 5)
        assert m.pixel_count > c1                                    # המטמון מתבטל בשינוי

    def test_bad_size(self):
        with pytest.raises(InvalidInputError):
            MaskLayer(0, 10)


# ------------------------------------------------------------------------------------------- Patch / History
def make_patch(history, rgb, em, color, alpha=None, a_after=None):
    ys, xs = np.nonzero(em)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    sub = em[box[1]:box[3], box[0]:box[2]] > 0
    before = rgb[box[1]:box[3], box[0]:box[2]][sub]
    after = np.tile(np.array(color, np.uint8), (before.shape[0], 1))
    ab = aa = None
    if alpha is not None:
        ab = alpha[box[1]:box[3], box[0]:box[2]][sub]
        aa = np.full(ab.shape, a_after, np.uint8)
    return Patch.create(history.next_id(), box, rgb.shape[:2], sub, before, after, ab, aa)


class TestPatch:
    def test_round_trip_and_input_untouched(self):
        rgb = rnd()
        em = np.zeros(rgb.shape[:2], np.uint8)
        em[10:40, 20:50] = 1
        em[15:25, 30:40] = 0                                          # חור: הפיקסלים שבו לא ישתנו
        h = History()
        p = make_patch(h, rgb, em, (1, 2, 3))
        orig = rgb.copy()
        fwd, a = p.apply(rgb, None, forward=True)
        assert a is None and np.array_equal(rgb, orig)               # הקלט לא השתנה
        assert (fwd[em > 0] == (1, 2, 3)).all() and np.array_equal(fwd[em == 0], rgb[em == 0])
        back, _ = p.apply(fwd, None, forward=False)
        assert np.array_equal(back, orig)

    def test_only_masked_pixels_stored(self):
        rgb = rnd(400, 400)
        em = np.zeros(rgb.shape[:2], np.uint8)
        em[0:400, 0:2] = 1                                            # פס צר לאורך כל המלבן
        em[0:2, 0:400] = 1                                            # ... ו-L: המלבן הדק הוא כל התמונה
        p = make_patch(History(), rgb, em, (7, 7, 7))
        assert p.count == int(em.sum()) and p.box == (0, 0, 400, 400)
        assert p.nbytes < 0.1 * rgb.nbytes                            # לא נשמר המלבן, וגם זה דחוס

    def test_alpha_round_trip(self):
        rgb, alpha = rnd(), rnd(seed=5)[..., 0].copy()
        em = box_mask(*rgb.shape[:2], 20, 60, 30, 90)
        p = make_patch(History(), rgb, em, (9, 9, 9), alpha, 255)
        f_rgb, f_a = p.apply(rgb, alpha, forward=True)
        assert (f_a[em > 0] == 255).all() and np.array_equal(f_a[em == 0], alpha[em == 0])
        b_rgb, b_a = p.apply(f_rgb, f_a, forward=False)
        assert np.array_equal(b_rgb, rgb) and np.array_equal(b_a, alpha)

    def test_mismatches_rejected(self):
        rgb = rnd()
        em = box_mask(*rgb.shape[:2], 5, 15, 5, 15)
        p = make_patch(History(), rgb, em, (1, 1, 1))
        with pytest.raises(InvalidInputError):
            p.apply(rnd(50, 50), None, forward=True)                 # גודל אחר
        with pytest.raises(InvalidInputError):
            p.apply(rgb, np.zeros(rgb.shape[:2], np.uint8), forward=True)   # Alpha שלא היה
        pa = make_patch(History(), rgb, em, (1, 1, 1), np.zeros(rgb.shape[:2], np.uint8), 0)
        with pytest.raises(InvalidInputError):
            pa.apply(rgb, None, forward=True)                         # Alpha חסר

    def test_create_validation(self):
        with pytest.raises(InvalidInputError):
            Patch.create(1, (0, 0, 4, 4), (10, 10), np.ones((4, 4), bool), np.zeros((3, 3), np.uint8),
                         np.zeros((3, 3), np.uint8))
        with pytest.raises(InvalidInputError):
            Patch.create(1, (0, 0, 2, 2), (10, 10), np.ones((2, 2), bool), np.zeros((4, 3), np.uint8),
                         np.zeros((4, 3), np.uint8), np.zeros(4, np.uint8), None)


class TestHistory:
    def _p(self, h, rgb, n=1, size=4):
        em = box_mask(*rgb.shape[:2], 0, size, 0, size)
        return make_patch(h, rgb, em, (n, n, n))

    def test_step_limit_drops_oldest(self):
        h, rgb = History(max_steps=3), rnd()
        ids = []
        for i in range(5):
            p = self._p(h, rgb, i)
            ids.append(p.id)
            h.push(p)
        assert h.undo_count == 3 and h.dropped_total == 2 and h.peek_undo().id == ids[-1]

    def test_byte_limit_keeps_newest_even_if_alone_over_limit(self):
        rgb = rnd(200, 200)
        h = History(max_steps=100, max_bytes=100)
        for i in range(4):
            h.push(self._p(h, rgb, i, size=50))
        assert h.undo_count == 1 and h.peek_undo().nbytes > 100      # נשאר הצעד האחרון בלבד

    def test_byte_limit_partial(self):
        rgb = np.zeros((200, 200, 3), np.uint8)
        h = History(max_steps=100)
        p = self._p(h, rgb, 1, size=50)
        h.max_bytes = int(p.nbytes * 2.5)
        for _ in range(5):
            h.push(self._p(h, rgb, 1, size=50))
        assert h.undo_count == 2 and h.nbytes <= h.max_bytes

    def test_redo_cleared_on_push_and_ids(self):
        h, rgb = History(), rnd()
        a, b = self._p(h, rgb, 1), self._p(h, rgb, 2)
        h.push(a)
        h.push(b)
        assert h.current_id == b.id
        h.mark_undone()
        assert h.can_redo and h.current_id == a.id and h.redo_count == 1
        h.push(self._p(h, rgb, 3))
        assert not h.can_redo
        h.mark_undone()
        h.mark_undone()
        assert h.current_id == 0 and not h.can_undo

    def test_invalid_limits(self):
        with pytest.raises(ValueError):
            History(max_steps=0)
        with pytest.raises(ValueError):
            History(max_bytes=0)


# ------------------------------------------------------------------------------------------- Session.commit
class TestCommit:
    def test_commit_changes_only_mask_new_arrays_frozen(self):
        s = session()
        before = s.rgb
        em = box_mask(120, 160, 30, 60, 40, 90)
        p = s.commit(fake_result(s.rgb, em, (200, 10, 10)))
        assert p is not None and s.version == 1 and s.rgb is not before
        assert (s.rgb[em > 0] == (200, 10, 10)).all()
        assert np.array_equal(s.rgb[em == 0], before[em == 0])
        assert not s.rgb.flags.writeable and not before.flags.writeable
        assert s.original_rgb is before and not np.array_equal(s.rgb, s.original_rgb)
        with pytest.raises(ValueError):
            s.rgb[0, 0, 0] = 1

    def test_no_change_is_noop(self):
        s = session()
        em = box_mask(120, 160, 10, 20, 10, 20)
        r = fake_result(s.rgb, em, (0, 0, 0))
        r = InpaintResult(s.rgb.copy(), em, r.roi)                    # תוצאה זהה למקור
        assert s.commit(r) is None and s.version == 0 and not s.history.can_undo and not s.modified

    def test_undo_redo_exact(self):
        s = session()
        orig = s.rgb.copy()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 50, 5, 50), (1, 2, 3)))
        c1 = s.rgb.copy()
        assert s.undo() is HistoryAction.UNDONE and np.array_equal(s.rgb, orig)
        assert s.redo() is HistoryAction.REDONE and np.array_equal(s.rgb, c1)
        assert s.undo() is HistoryAction.UNDONE and s.undo() is HistoryAction.NOTHING
        assert s.redo() is HistoryAction.REDONE and s.redo() is HistoryAction.NOTHING

    def test_twenty_steps_undo_redo(self):
        """DoD של M3: 20 צעדי Undo/Redo, כולל משיכות חופפות."""
        s = session(200, 240)
        states = [s.rgb.copy()]
        rng = np.random.default_rng(1)
        for i in range(20):
            y, x = int(rng.integers(0, 150)), int(rng.integers(0, 180))
            em = box_mask(200, 240, y, y + 40, x, x + 50)
            s.commit(fake_result(s.rgb, em, tuple(int(v) for v in rng.integers(0, 256, 3))))
            states.append(s.rgb.copy())
        assert s.history.undo_count == 20
        for i in range(20, 0, -1):
            assert s.undo() is HistoryAction.UNDONE and np.array_equal(s.rgb, states[i - 1])
        for i in range(1, 21):
            assert s.redo() is HistoryAction.REDONE and np.array_equal(s.rgb, states[i])
        assert np.array_equal(s.original_rgb, states[0])

    def test_new_commit_clears_redo(self):
        s = session()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        s.undo()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 50, 70, 50, 70), (2, 2, 2)))
        assert s.redo() is HistoryAction.NOTHING

    def test_modified_flag_and_save(self, tmp_path):
        s = session()
        assert not s.modified
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        assert s.modified
        s.undo()
        assert not s.modified                                          # חזרה למצב הנטען
        s.redo()
        assert s.modified
        res = s.save(tmp_path / "תמונה שמורה.png")
        assert not s.modified and res.path.exists()
        back = load_image(res.path)
        assert np.array_equal(back.rgb, s.rgb)
        s.commit(fake_result(s.rgb, box_mask(120, 160, 50, 70, 50, 70), (2, 2, 2)))
        s.undo()
        assert not s.modified                                          # חזרה למצב השמור
        s.undo()
        assert s.modified                                              # מתחת למצב השמור

    def test_save_failure_keeps_modified(self, tmp_path):
        s = session()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        with pytest.raises(FixPicError):
            s.save(tmp_path / "אין" / "x.png")
        assert s.modified

    def test_overlapping_stale_results_keep_earlier_commit(self):
        """שני Jobs שהוגשו על אותו בסיס: ה-Commit של השני לא מוחק את הראשון (הדבקה לפי מסכה, לא החלפת תמונה)."""
        s = session()
        base = s.rgb
        r1 = fake_result(base, box_mask(120, 160, 10, 40, 10, 60), (10, 10, 10))
        r2 = fake_result(base, box_mask(120, 160, 30, 70, 40, 100), (20, 20, 20))
        s.commit(r1)
        s.commit(r2)
        assert (s.rgb[10:30, 10:60] == 10).all()                      # החלק הלא חופף של הראשון שרד
        assert (s.rgb[30:70, 40:100] == 20).all()                      # בחפיפה השני גובר
        s.undo()
        assert (s.rgb[30:40, 40:60] == 10).all()                      # Undo של השני מחזיר את מצב הראשון
        s.undo()
        assert np.array_equal(s.rgb, base)

    def test_invalid_results_rejected_state_unchanged(self):
        s = session()
        ok = box_mask(120, 160, 5, 20, 5, 20)
        for res in (fake_result(rnd(50, 50), box_mask(50, 50, 1, 5, 1, 5), (1, 1, 1)),        # גודל אחר
                    fake_result(s.rgb, np.zeros((120, 160), np.uint8), (1, 1, 1))):          # מסכה ריקה
            with pytest.raises(FixPicError):
                s.commit(res)
        assert s.version == 0 and not s.history.can_undo
        s.commit(fake_result(s.rgb, ok, (3, 3, 3)))                   # ועדיין עובד

    def test_commit_is_atomic_on_memory_error(self, monkeypatch):
        s = session()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        snap, ver, undo_n = s.rgb, s.version, s.history.undo_count

        def boom(*a, **k):
            raise MemoryError("simulated")
        monkeypatch.setattr(session_mod.Patch, "apply", boom)
        with pytest.raises(OutOfMemoryError):
            s.commit(fake_result(s.rgb, box_mask(120, 160, 50, 70, 50, 70), (9, 9, 9)))
        assert s.rgb is snap and s.version == ver and s.history.undo_count == undo_n

    def test_undo_memory_error_leaves_state(self, monkeypatch):
        s = session()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        snap = s.rgb
        monkeypatch.setattr(session_mod.Patch, "apply", lambda *a, **k: (_ for _ in ()).throw(MemoryError("x")))
        with pytest.raises(OutOfMemoryError):
            s.undo()
        assert s.rgb is snap and s.history.can_undo and not s.history.can_redo

    def test_history_trim_still_undoes_recent(self):
        s = session(max_steps=3)
        states = [s.rgb.copy()]
        for i in range(6):
            s.commit(fake_result(s.rgb, box_mask(120, 160, i * 10, i * 10 + 8, 5, 30), (i + 1, i, i)))
            states.append(s.rgb.copy())
        assert s.history.undo_count == 3 and s.history.dropped_total == 3
        for i in (5, 4, 3):
            s.undo()
            assert np.array_equal(s.rgb, states[i])
        assert s.undo() is HistoryAction.NOTHING
        assert s.original_rgb is not None and np.array_equal(s.original_rgb, states[0])

    def test_validation_of_input_image(self):
        li = image_from_array(rnd())
        li.alpha = np.zeros((5, 5), np.uint8)
        with pytest.raises(InvalidInputError):
            Session(li)
        li = image_from_array(rnd())
        li.rgb = li.rgb[..., :1]
        with pytest.raises(InvalidInputError):
            Session(li)


# ------------------------------------------------------------------------------------------- Alpha
class TestAlpha:
    def test_opaque_stays_opaque_and_transparent_stays_transparent(self):
        h, w = 120, 160
        alpha = np.full((h, w), 255, np.uint8)
        alpha[:, 80:] = 0                                              # חצי שמאלי אטום, חצי ימני שקוף
        s = session(h, w, alpha=alpha)
        s.commit(fake_result(s.rgb, box_mask(h, w, 20, 50, 20, 50), (5, 5, 5)))       # בתוך האטום
        s.commit(fake_result(s.rgb, box_mask(h, w, 60, 90, 100, 140), (6, 6, 6)))     # בתוך השקוף
        assert (s.alpha[20:50, 20:50] == 255).all()
        assert (s.alpha[60:90, 100:140] == 0).all()
        assert np.array_equal(s.alpha[:20], alpha[:20])                # מחוץ למסכות לא השתנה

    def test_soft_edge_is_continued(self):
        h, w = 100, 200
        alpha = np.tile(np.linspace(0, 255, w).astype(np.uint8), (h, 1))
        s = session(h, w, alpha=alpha)
        s.commit(fake_result(s.rgb, box_mask(h, w, 30, 70, 80, 120), (1, 1, 1)))
        got = s.alpha[30:70, 80:120].astype(int)
        want = alpha[30:70, 80:120].astype(int)
        err = np.abs(got - want)
        assert err.mean() < 3 and err.max() <= 10                  # המשכיות הגרדיאנט (נמדד: ממוצע 1.9, מקסימום 7)

    def test_undo_restores_alpha_exactly(self):
        h, w = 120, 160
        alpha = np.random.default_rng(2).integers(0, 256, (h, w), dtype=np.uint8)
        s = session(h, w, alpha=alpha)
        s.commit(fake_result(s.rgb, box_mask(h, w, 20, 60, 30, 90), (5, 5, 5)))
        assert not np.array_equal(s.alpha, alpha)
        s.undo()
        assert np.array_equal(s.alpha, alpha) and s.alpha is not None
        s.redo()
        assert not np.array_equal(s.alpha, alpha)

    def test_no_alpha_stays_none(self):
        s = session()
        s.commit(fake_result(s.rgb, box_mask(120, 160, 5, 20, 5, 20), (1, 1, 1)))
        assert s.alpha is None and s.original_alpha is None

    def test_mask_covering_everything_gives_opaque(self):
        h, w = 40, 50
        alpha = np.zeros((h, w), np.uint8)
        s = session(h, w, alpha=alpha)
        s.commit(fake_result(s.rgb, np.ones((h, w), np.uint8), (5, 5, 5)))
        assert (s.alpha == 255).all()

    def test_fill_alpha_large_region_bounded_time_and_input_untouched(self):
        h, w = 1500, 1500
        alpha = np.full((h, w), 255, np.uint8)
        alpha[:, :700] = 0
        mask = box_mask(h, w, 200, 1300, 300, 1300)
        sub = mask[200:1300, 300:1300] > 0
        before = alpha.copy()
        t = time.perf_counter()
        out = fill_alpha(alpha, mask, (300, 200, 1300, 1300), sub)
        assert time.perf_counter() - t < 2.0
        assert out.shape == (int(sub.sum()),) and np.array_equal(alpha, before)


# ------------------------------------------------------------------------------------------- אינטגרציה עם Scheduler
def gate_session(gated=True, h=120, w=120, **kw):
    model = GateModel(gated=gated)
    sched = JobScheduler(Pipeline(model, PipelineConfig(mask_dilate_px=0)))
    sched.start(warmup=False)
    li = image_from_array(np.full((h, w, 3), 50, np.uint8))
    return model, sched, Session(li, sched, **kw)


def stroke(s, x=60, y=60, d=20):
    s.mask.paint([(x, y)], d)


class TestSchedulerIntegration:
    def test_stroke_job_commit_cycle(self):
        model, sched, s = gate_session(gated=False)
        with sched:
            stroke(s)
            assert not s.busy
            job = s.submit_stroke(size=256)
            assert s.busy and s.mask.empty and len(s.pending_masks) == 1 and s.pending_masks[0].any()
            assert job.wait(T)
            out = s.finish(job)
            assert out.kind is OutcomeKind.COMMITTED and out.patch is not None
            assert not s.busy and s.pending_masks == [] and s.mask.empty
            assert (s.rgb[55:65, 55:65] == 9).all() and (s.rgb[0:10, 0:10] == 50).all()
            assert s.undo() is HistoryAction.UNDONE and (s.rgb == 50).all()

    def test_brush_px_flows_to_job(self):
        _, sched, s = gate_session(gated=False)
        with sched:
            stroke(s, d=33)
            job = s.submit_stroke(size=256)
            assert job.brush_px == 33.0
            job.wait(T)
            s.finish(job)
            stroke(s, d=10)
            job = s.submit_stroke(size=256, brush_px=7)
            assert job.brush_px == 7.0
            job.wait(T)
            s.finish(job)

    def test_second_stroke_runs_on_updated_image_when_serial(self):
        _, sched, s = gate_session(gated=False)
        with sched:
            stroke(s, 30, 30)
            j1 = s.submit_stroke(size=256)
            j1.wait(T)
            s.finish(j1)
            stroke(s, 90, 90)
            j2 = s.submit_stroke(size=256)
            j2.wait(T)
            s.finish(j2)
            assert (s.rgb[25:35, 25:35] == 9).all() and (s.rgb[85:95, 85:95] == 9).all()
            assert s.history.undo_count == 2

    def test_queued_strokes_both_survive_commit(self):
        """משיכה שנייה הוגשה על התמונה שלפני ה-Commit של הראשונה (תור): שתיהן נשארות (הדבקה לפי מסכה)."""
        model, sched, s = gate_session()
        with sched:
            stroke(s, 30, 30)
            j1 = s.submit_stroke(size=256)
            assert model.entered.acquire(timeout=T)
            stroke(s, 90, 90)
            j2 = s.submit_stroke(size=256)
            assert len(s.pending_masks) == 2
            model.gate.release()
            assert j1.wait(T)
            assert s.finish(j1).kind is OutcomeKind.COMMITTED
            assert model.entered.acquire(timeout=T)
            model.gate.release()
            assert j2.wait(T)
            assert s.finish(j2).kind is OutcomeKind.COMMITTED
            assert (s.rgb[25:35, 25:35] == 9).all() and (s.rgb[85:95, 85:95] == 9).all()

    def test_empty_mask_and_no_scheduler(self):
        _, sched, s = gate_session(gated=False)
        with sched:
            with pytest.raises(EmptyMaskError):
                s.submit_stroke()
        with pytest.raises(InvalidInputError):
            session().submit_stroke()

    def test_queue_full_keeps_mask(self):
        model, sched, s = gate_session()
        with sched:
            jobs = []
            stroke(s)
            jobs.append(s.submit_stroke(size=256))
            assert model.entered.acquire(timeout=T)
            for _ in range(3):
                stroke(s)
                jobs.append(s.submit_stroke(size=256))
            stroke(s, 20, 20)
            with pytest.raises(QueueFullError):
                s.submit_stroke(size=256)
            assert not s.mask.empty and len(s.pending_masks) == 4      # המסכה נשארה, והמשיכות בעיבוד לא נפגעו
            assert s.undo() is HistoryAction.CANCELLED_JOBS
            model.gate.release()

    def test_failure_restores_mask_and_image_untouched(self):
        class Boom(GateModel):
            def run(self, image, mask, cancel=None):
                raise RuntimeError("backend died")
        sched = JobScheduler(Pipeline(Boom(gated=False), PipelineConfig(mask_dilate_px=0)))
        sched.start(warmup=False)
        s = Session(image_from_array(np.full((120, 120, 3), 50, np.uint8)), sched)
        with sched:
            stroke(s)
            snap_mask = s.mask.snapshot()
            job = s.submit_stroke(size=256)
            assert job.wait(T)
            out = s.finish(job)
            assert out.kind is OutcomeKind.FAILED and isinstance(out.error, InferenceError)
            assert np.array_equal(s.mask.snapshot(), snap_mask)        # PARITY-SPEC §3.6: המסכה נשארת
            assert (s.rgb == 50).all() and not s.history.can_undo and s.version == 0

    def test_failed_restore_merges_with_new_strokes(self):
        class Boom(GateModel):
            def run(self, image, mask, cancel=None):
                raise RuntimeError("x")
        sched = JobScheduler(Pipeline(Boom(gated=False), PipelineConfig(mask_dilate_px=0)))
        sched.start(warmup=False)
        s = Session(image_from_array(np.full((120, 120, 3), 50, np.uint8)), sched)
        with sched:
            stroke(s, 30, 30)
            job = s.submit_stroke(size=256)
            stroke(s, 90, 90)                                           # נצייר משהו חדש לפני הכשל
            job.wait(T)
            s.finish(job)
            assert s.mask.array[30, 30] == 255 and s.mask.array[90, 90] == 255

    def test_direct_cancel_restores_mask(self):
        model, sched, s = gate_session()
        with sched:
            stroke(s)
            job = s.submit_stroke(size=256)
            assert model.entered.acquire(timeout=T)
            assert job.cancel()
            model.gate.release()
            assert job.wait(T) and job.state is JobState.CANCELLED
            out = s.finish(job)
            assert out.kind is OutcomeKind.CANCELLED and not s.mask.empty and (s.rgb == 50).all()

    def test_undo_during_processing_cancels_everything(self):
        model, sched, s = gate_session()
        with sched:
            # היסטוריה קיימת, כדי להוכיח שה-Undo בזמן עיבוד אינו נוגע בה
            s.commit(fake_result(s.rgb, box_mask(120, 120, 5, 15, 5, 15), (77, 77, 77)))
            committed = s.rgb
            stroke(s, 30, 30)
            j1 = s.submit_stroke(size=256)
            assert model.entered.acquire(timeout=T)
            stroke(s, 90, 90)
            j2 = s.submit_stroke(size=256)
            assert s.undo() is HistoryAction.CANCELLED_JOBS
            assert not s.busy and s.pending_masks == []
            model.gate.release()
            assert j1.wait(T) and j2.wait(T)
            assert j1.state is JobState.CANCELLED and j2.state is JobState.CANCELLED
            assert s.finish(j1).kind is OutcomeKind.STALE and s.finish(j2).kind is OutcomeKind.STALE
            assert s.rgb is committed and s.history.undo_count == 1 and s.mask.empty
            assert s.undo() is HistoryAction.UNDONE                       # עכשיו ה-Undo הבא פועל על ההיסטוריה

    def test_race_job_done_but_not_yet_finished_when_undo(self):
        """ה-Job הסתיים (DONE), ה-UI טרם קלט אותו, ובינתיים Undo: התוצאה אסור שתיכתב."""
        _, sched, s = gate_session(gated=False)
        with sched:
            stroke(s)
            job = s.submit_stroke(size=256)
            assert job.wait(T) and job.state is JobState.DONE
            assert s.undo() is HistoryAction.CANCELLED_JOBS
            assert s.finish(job).kind is OutcomeKind.STALE
            assert (s.rgb == 50).all() and not s.history.can_undo

    def test_redo_busy_and_finish_unfinished(self):
        model, sched, s = gate_session()
        with sched:
            s.commit(fake_result(s.rgb, box_mask(120, 120, 5, 15, 5, 15), (77, 77, 77)))
            s.undo()
            stroke(s)
            job = s.submit_stroke(size=256)
            assert model.entered.acquire(timeout=T)
            assert s.redo() is HistoryAction.BUSY and s.history.can_redo   # Redo נדחה ולא אבד
            with pytest.raises(InvalidInputError):
                s.finish(job)
            assert s.busy                                                   # אחרי השגיאה ה-Job עדיין רשום
            model.gate.release()
            job.wait(T)
            assert s.finish(job).kind is OutcomeKind.COMMITTED
            assert not s.history.can_redo                                   # Commit חדש איפס Redo

    def test_finish_unknown_job_is_stale(self):
        _, sched, s = gate_session(gated=False)
        _, sched2, s2 = gate_session(gated=False)
        with sched, sched2:
            stroke(s2)
            job = s2.submit_stroke(size=256)
            job.wait(T)
            assert s.finish(job).kind is OutcomeKind.STALE

    def test_callbacks_run_and_session_state_only_changes_in_finish(self):
        _, sched, s = gate_session(gated=False)
        done = threading.Event()
        with sched:
            stroke(s)
            job = s.submit_stroke(size=256, on_done=lambda j: done.set())
            assert done.wait(T)
            assert (s.rgb == 50).all()                                      # עד ה-finish התמונה לא השתנתה
            s.finish(job)
            assert (s.rgb[55:65, 55:65] == 9).all()

    def test_working_image_is_readonly_while_jobs_hold_it(self):
        _, sched, s = gate_session(gated=False)
        with sched:
            stroke(s)
            job = s.submit_stroke(size=256)
            job.wait(T)
            assert not s.rgb.flags.writeable
            s.finish(job)


# ------------------------------------------------------------------------------------------- מקצה לקצה עם Pipeline אמיתי
def test_end_to_end_real_pipeline_with_alpha(tiny):
    """Pipeline אמיתי (מודל ONNX סינתטי, ORT) דרך Scheduler ו-Session, כולל Alpha, Undo ו-Redo."""
    from fixpic.models import AutoModel, ModelConfig
    model = AutoModel()
    model.load(ModelConfig(tiny, threads=2))
    rgb = rnd(300, 400, seed=11)
    alpha = np.full((300, 400), 255, np.uint8)
    alpha[:, 250:] = 128
    li = image_from_array(rgb)
    li.alpha = alpha
    with JobScheduler(Pipeline(model, PipelineConfig(mask_dilate_px=2))).start(warmup=False) as sched:
        s = Session(li, sched)
        s.mask.paint([(120, 150), (200, 150)], 30)
        job = s.submit_stroke(size=256)
        assert job.wait(30)
        out = s.finish(job)
        assert out.kind is OutcomeKind.COMMITTED, out.error
        eff = job.result().effective_mask > 0
        assert np.array_equal(s.rgb[~eff], rgb[~eff])                   # מחוץ למסכה זהה למקור
        assert not np.array_equal(s.rgb[eff], rgb[eff])
        assert (s.alpha[eff] == 255).all()                              # סביבה אטומה
        s.undo()
        assert np.array_equal(s.rgb, rgb) and np.array_equal(s.alpha, alpha)
        s.redo()
        assert not np.array_equal(s.rgb, rgb)
