"""M4: `MaskHistory` (Undo/Redo לעריכות מסכה, ללא UI) ו-`MaskLayer.apply_diff`."""
from __future__ import annotations

import time

import numpy as np
import pytest

from fixpic.errors import InvalidInputError
from fixpic.session import MaskHistory, MaskLayer

TRI = [(10, 10), (50, 10), (30, 60)]


def setup(h=100, w=120, **kw):
    ml = MaskLayer(h, w)
    return ml, MaskHistory(ml, **kw)


class TestBasics:
    def test_do_undo_redo_roundtrip_each_tool(self):
        ml, hist = setup()
        ops = [
            lambda m: m.paint([(10, 80), (100, 80)], 7),
            lambda m: m.fill_polygon(TRI),
            lambda m: m.grow(3),
            lambda m: m.shrink(1),
            lambda m: m.fill_polygon([(5, 5), (30, 5), (20, 25)], erase=True),
            lambda m: m.remove_component_at(100, 80),
        ]
        states = [ml.snapshot()]
        for op in ops:
            hist.do(op)
            states.append(ml.snapshot())
        assert hist.undo_count == len(ops)
        for i in range(len(ops), 0, -1):
            assert hist.undo() is not None
            assert np.array_equal(ml.array, states[i - 1])
        assert not hist.can_undo and hist.undo() is None
        for i in range(1, len(ops) + 1):
            assert hist.redo() is not None
            assert np.array_equal(ml.array, states[i])
        assert not hist.can_redo and hist.redo() is None

    def test_do_returns_op_result(self):
        ml, hist = setup()
        assert hist.do(lambda m: m.fill_polygon(TRI)) is not None
        assert hist.do(lambda m: m.fill_polygon(TRI)) is None      # no-op: לא נשמר צעד
        assert hist.undo_count == 1

    def test_noop_ops_do_not_record(self):
        ml, hist = setup()
        hist.do(lambda m: m.grow(3))                  # מסכה ריקה
        hist.do(lambda m: m.fill_polygon([(5, 5), (9, 9), (20, 20)]))   # שטח אפס
        hist.do(lambda m: m.remove_small(5))
        assert hist.undo_count == 0 and not hist.stale

    def test_undo_returns_rect_of_change_and_is_tight(self):
        ml, hist = setup()
        hist.do(lambda m: m.fill_polygon([(10, 10), (50, 10), (50, 40), (10, 40)]))
        assert hist.undo() == (10, 10, 51, 41)

    def test_redo_cleared_by_new_step(self):
        ml, hist = setup()
        hist.do(lambda m: m.fill_polygon(TRI))
        hist.undo()
        assert hist.can_redo
        hist.do(lambda m: m.paint([(5, 90)], 5))
        assert not hist.can_redo and hist.undo_count == 1

    def test_brush_px_restored(self):
        ml, hist = setup()
        hist.do(lambda m: m.paint([(10, 10), (20, 20)], 9))
        hist.do(lambda m: m.paint([(40, 40)], 30))
        assert ml.brush_px == 30
        hist.undo()
        assert ml.brush_px == 9
        hist.undo()
        assert ml.brush_px is None and ml.empty
        hist.redo(); hist.redo()
        assert ml.brush_px == 30

    def test_shrink_to_empty_resets_brush_and_undo_restores(self):
        ml, hist = setup()
        ml.paint([(30, 30)], 5)
        hist.clear()
        hist.do(lambda m: m.shrink(10))
        assert ml.empty and ml.brush_px is None
        hist.undo()
        assert not ml.empty and ml.brush_px == 5


class TestStrokes:
    def test_begin_end_groups_many_paints_in_one_step(self):
        ml, hist = setup()
        hist.begin()
        for x in range(10, 100, 10):
            ml.paint([(x, 20), (x + 10, 25)], 6)
        assert hist.recording and hist.end() is True
        assert hist.undo_count == 1
        hist.undo()
        assert ml.empty

    def test_cancel_restores_state_and_brush(self):
        ml, hist = setup()
        ml.paint([(10, 10)], 4)
        hist.clear()
        before = ml.snapshot()
        hist.begin()
        ml.paint([(60, 60), (90, 90)], 20)
        assert hist.end(cancel=True) is False
        assert np.array_equal(ml.array, before) and ml.brush_px == 4 and hist.undo_count == 0 and not hist.stale

    def test_exception_in_op_rolls_back(self):
        ml, hist = setup()
        ml.paint([(10, 10)], 4)
        hist.clear()
        before = ml.snapshot()

        def bad(m):
            m.fill_polygon(TRI)
            raise RuntimeError("boom")
        with pytest.raises(RuntimeError):
            hist.do(bad)
        assert np.array_equal(ml.array, before) and not hist.recording and hist.undo_count == 0

    def test_begin_twice_and_end_without_begin(self):
        ml, hist = setup()
        with pytest.raises(InvalidInputError):
            hist.end()
        hist.begin()
        with pytest.raises(InvalidInputError):
            hist.begin()
        with pytest.raises(InvalidInputError):
            hist.undo()
        hist.end()

    def test_no_change_stroke_not_recorded(self):
        ml, hist = setup()
        hist.begin()
        assert hist.end() is False and hist.undo_count == 0


class TestStale:
    def test_external_change_makes_history_stale(self):
        ml, hist = setup()
        hist.do(lambda m: m.fill_polygon(TRI))
        ml.take()                                   # כמו Session.submit_stroke
        assert hist.stale and not hist.can_undo and hist.undo() is None and hist.undo_count == 0
        assert ml.empty                              # undo מיושן לא נגע בשכבה

    def test_next_op_clears_stale_history_and_records_again(self):
        ml, hist = setup()
        hist.do(lambda m: m.fill_polygon(TRI))
        ml.add(np.full((100, 120), 255, np.uint8))   # שחזור אחרי כשל
        hist.do(lambda m: m.fill_polygon(TRI, erase=True))
        assert not hist.stale and hist.undo_count == 1
        hist.undo()
        assert (ml.array == 255).all()
        assert hist.undo() is None                   # הצעד הישן נזרק, לא הוחל על מסכה אחרת

    def test_clear_resyncs(self):
        ml, hist = setup()
        hist.do(lambda m: m.fill_polygon(TRI))
        ml.clear()
        hist.clear()
        assert not hist.stale and hist.undo_count == 0


class TestLimits:
    def test_max_steps_drops_oldest(self):
        ml, hist = setup(max_steps=3)
        for i in range(6):
            hist.do(lambda m, i=i: m.paint([(10 + 15 * i, 50)], 5))
        assert hist.undo_count == 3 and hist.dropped_total == 3
        for _ in range(3):
            assert hist.undo() is not None
        assert hist.undo() is None and ml.pixel_count > 0      # שלושת הציורים הראשונים נשארו

    def test_max_bytes_keeps_last_step_even_if_over(self):
        ml, hist = setup(max_bytes=100)
        hist.do(lambda m: m.fill_polygon([(0, 0), (99, 0), (99, 99), (0, 99)]))
        assert hist.undo_count == 1 and hist.nbytes > 100
        hist.do(lambda m: m.paint([(60, 60)], 40))
        assert hist.undo_count == 1

    def test_invalid_limits(self):
        ml = MaskLayer(10, 10)
        with pytest.raises(ValueError):
            MaskHistory(ml, max_steps=0)
        with pytest.raises(ValueError):
            MaskHistory(ml, max_bytes=0)

    def test_memory_is_by_change_size_not_image(self):
        ml = MaskLayer(4000, 6000)
        hist = MaskHistory(ml)
        t0 = time.perf_counter()
        hist.do(lambda m: m.fill_polygon([(100, 100), (200, 100), (150, 200)]))
        assert hist.nbytes < 20_000 and time.perf_counter() - t0 < 1.0
        t0 = time.perf_counter()
        hist.undo(); hist.redo()
        assert time.perf_counter() - t0 < 0.1


class TestApplyDiff:
    def test_invalid_diff(self):
        ml = MaskLayer(20, 30)
        with pytest.raises(InvalidInputError):
            ml.apply_diff((0, 0, 5, 5), np.zeros((4, 5), np.uint8), None)
        with pytest.raises(InvalidInputError):
            ml.apply_diff((0, 0, 40, 5), np.zeros((5, 40), np.uint8), None)
        with pytest.raises(InvalidInputError):
            ml.apply_diff((0, 0, 5, 5), np.zeros((5, 5), np.float32), None)
        assert ml.version == 0

    def test_xor_toggles_and_bumps_version(self):
        ml = MaskLayer(20, 30)
        d = np.full((5, 5), 255, np.uint8)
        ml.apply_diff((2, 2, 7, 7), d, 3.0)
        assert ml.pixel_count == 25 and ml.brush_px == 3.0 and ml.version == 1
        ml.apply_diff((2, 2, 7, 7), d, None)
        assert ml.empty and ml.brush_px is None
