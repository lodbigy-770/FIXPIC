"""M4 (לאסו, שירות ללא UI): `mask_ops.polygon_region` ו-`MaskLayer.fill_polygon`."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from fixpic.errors import InvalidInputError
from fixpic.session import MaskLayer, mask_ops


def full(points, shape=(100, 120)):
    """המצולע כמסכה בגודל התמונה (או None)."""
    r = mask_ops.polygon_region(points, shape)
    if r is None:
        return None
    reg, (x0, y0, x1, y1) = r
    out = np.zeros(shape, np.uint8)
    out[y0:y1, x0:x1] = reg
    return out


def ref_fill(points, shape=(100, 120)):
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [np.array(points, np.int32).reshape(-1, 1, 2)], 255)
    return out


def star(cx, cy, r_out, r_in, n, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(2 * n):
        a = np.pi * i / n
        r = (r_out if i % 2 == 0 else r_in) * float(rng.uniform(0.8, 1.2))
        pts.append((round(cx + r * np.cos(a)), round(cy + r * np.sin(a))))
    return pts


class TestPolygonRegion:
    def test_rectangle_is_exact_and_inclusive(self):
        m = full([(10, 10), (50, 10), (50, 40), (10, 40)])
        assert np.array_equal(m, ref_fill([(10, 10), (50, 10), (50, 40), (10, 40)]))
        assert int((m > 0).sum()) == 41 * 31

    def test_triangle_matches_cv2(self):
        pts = [(5, 5), (90, 20), (30, 80)]
        a, b = full(pts) > 0, ref_fill(pts) > 0
        assert (a ^ b).sum() <= 0.01 * b.sum()

    @pytest.mark.parametrize("seed", range(6))
    def test_random_simple_polygons_close_to_cv2(self, seed):
        pts = star(60, 50, 40, 20, 7, seed)
        a, b = full(pts) > 0, ref_fill(pts) > 0
        assert (a ^ b).sum() <= 0.02 * b.sum()

    def test_orientation_does_not_matter(self):
        pts = star(60, 50, 40, 20, 6, 3)
        assert np.array_equal(full(pts), full(pts[::-1]))

    def test_start_point_does_not_matter(self):
        pts = star(60, 50, 40, 20, 6, 4)
        assert np.array_equal(full(pts), full(pts[5:] + pts[:5]))

    def test_closing_point_repeated_is_same(self):
        pts = [(10, 10), (60, 15), (40, 70)]
        assert np.array_equal(full(pts), full(pts + [pts[0]]))

    def test_double_loop_is_filled_unlike_even_odd(self):
        # עיגול שהמסלול מקיף פעמיים: winding=2 בכל הפנים. cv2.fillPoly משאיר שם חור (even-odd)
        t = np.linspace(0, 4 * np.pi, 400, endpoint=False)
        pts = [(round(60 + 30 * np.cos(a)), round(50 + 30 * np.sin(a))) for a in t]
        m = full(pts) > 0
        assert m[50, 60] and m[50, 45] and m[60, 70]
        assert (ref_fill(pts) > 0).sum() < m.sum()

    def test_figure_eight_fills_both_lobes(self):
        pts = [(10, 10), (50, 50), (50, 10), (10, 50)]   # חוצה את עצמו במרכז
        m = full(pts, (60, 60)) > 0
        assert m[30, 15] and m[30, 45]                    # שתי האונות (שמאל וימין)
        assert not m[15, 30] and not m[45, 30]            # מעל ומתחת למרכז: מחוץ

    def test_collinear_and_zero_area_return_none(self):
        assert mask_ops.polygon_region([(5, 5), (30, 5), (60, 5)], (50, 80)) is None
        assert mask_ops.polygon_region([(5, 5), (30, 30), (60, 60)], (80, 80)) is None
        assert mask_ops.polygon_region([(7, 7), (7, 7), (7, 7)], (20, 20)) is None

    def test_fully_outside_returns_none(self):
        assert mask_ops.polygon_region([(200, 200), (300, 200), (250, 300)], (100, 120)) is None
        assert mask_ops.polygon_region([(-50, -50), (-10, -50), (-10, -10)], (100, 120)) is None

    def test_clipped_to_image_equals_clip_of_big_polygon(self):
        shape = (100, 120)
        big = [(-30, -20), (200, -20), (200, 150), (-30, 150)]
        assert np.all(full(big, shape) == 255)
        tri = [(-40, 50), (150, -30), (150, 130)]
        sub = full(tri, shape) > 0
        shifted = [(x + 100, y + 100) for x, y in tri]
        big_shape = (400, 420)
        whole = full(shifted, big_shape) > 0
        # חיתוך = החלון של מצולע שלם. הפנים זהים בדיוק; פיקסל שפה אחד על אלכסון יכול להשתנות, כי cv2.polylines חותך קווים
        w = whole[100:200, 100:220]
        k = np.ones((3, 3), np.uint8)
        edge = (cv2.dilate(sub.astype(np.uint8), k) - cv2.erode(sub.astype(np.uint8), k)) > 0
        assert not ((sub ^ w) & ~edge).any()
        assert (sub ^ w).sum() <= 0.5 * edge.sum()

    def test_window_is_inside_image_and_rect_matches(self):
        reg, rect = mask_ops.polygon_region([(-10, 20), (40, -5), (60, 70)], (50, 80))
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= 80 and 0 <= y0 < y1 <= 50
        assert reg.shape == (y1 - y0, x1 - x0) and reg.dtype == np.uint8 and set(np.unique(reg)) <= {0, 255}

    def test_float_points_and_numpy_input(self):
        pts = [(10.4, 10.6), (50.2, 10.1), (49.9, 40.4), (10.0, 39.7)]
        assert np.array_equal(full(pts), full(np.array(pts)))
        assert np.array_equal(full(pts), full([(10, 11), (50, 10), (50, 40), (10, 40)]))

    @pytest.mark.parametrize("bad", [
        [], [(1, 1)], [(1, 1), (2, 2)], [(1, 2, 3)] * 3, "abc", None, [(float("nan"), 1), (2, 2), (3, 1)],
        [(float("inf"), 1), (2, 2), (3, 1)], [(2_000_000, 1), (2, 2), (3, 1)], [(1, None), (2, 2), (3, 1)],
    ])
    def test_invalid_points(self, bad):
        with pytest.raises(InvalidInputError):
            mask_ops.polygon_region(bad, (50, 50))

    def test_too_many_points_and_bad_shape(self):
        pts = [(i % 50, (i * 7) % 50) for i in range(mask_ops.MAX_POLY_POINTS + 1)]
        with pytest.raises(InvalidInputError):
            mask_ops.polygon_region(pts, (50, 50))
        with pytest.raises(InvalidInputError):
            mask_ops.polygon_region([(1, 1), (5, 1), (3, 4)], (0, 10))

    def test_chunking_gives_same_result(self, monkeypatch):
        pts = star(60, 50, 45, 15, 40, 9)
        a = full(pts)
        monkeypatch.setattr(mask_ops, "_POLY_CHUNK_ROWS", 7)
        assert np.array_equal(a, full(pts))

    def test_input_not_modified(self):
        pts = np.array(star(60, 50, 40, 20, 6, 1), np.float64)
        orig = pts.copy()
        mask_ops.polygon_region(pts, (100, 120))
        assert np.array_equal(pts, orig)

    def test_large_image_small_lasso_is_cheap_and_large_lasso_is_ok(self):
        shape = (4000, 6000)    # 24MP
        t0 = time.perf_counter()
        r = mask_ops.polygon_region(star(3000, 2000, 80, 40, 12), shape)
        assert r is not None and r[0].shape[0] < 400 and time.perf_counter() - t0 < 0.5
        t0 = time.perf_counter()
        r = mask_ops.polygon_region([(0, 0), (5999, 0), (5999, 3999), (0, 3999)], shape)
        assert r is not None and r[0].shape == shape and (r[0] == 255).all()
        assert time.perf_counter() - t0 < 5.0


class TestFillPolygon:
    def test_fill_and_version_and_rect(self):
        ml = MaskLayer(100, 120)
        v = ml.version
        rect = ml.fill_polygon([(10, 10), (50, 10), (50, 40), (10, 40)])
        assert rect == (10, 10, 51, 41) and ml.version == v + 1
        assert ml.pixel_count == 41 * 31 and ml.bbox() == rect and ml.array.dtype == np.uint8

    def test_fill_is_union_not_replace(self):
        ml = MaskLayer(100, 120)
        ml.paint([(5, 80), (100, 80)], 6)
        before = ml.snapshot()
        ml.fill_polygon([(10, 10), (50, 10), (50, 40), (10, 40)])
        assert (ml.array[before > 0] == 255).all()

    def test_second_identical_fill_is_noop(self):
        ml = MaskLayer(100, 120)
        pts = [(10, 10), (50, 10), (30, 60)]
        ml.fill_polygon(pts)
        v = ml.version
        assert ml.fill_polygon(pts) is None and ml.version == v

    def test_erase_removes_only_polygon_pixels(self):
        ml = MaskLayer(100, 120)
        ml.add(np.full((100, 120), 255, np.uint8))
        rect = ml.fill_polygon([(20, 20), (60, 20), (60, 60), (20, 60)], erase=True)
        assert rect is not None
        a = ml.array
        assert (a[20:61, 20:61] == 0).all() and a.sum() // 255 == 100 * 120 - 41 * 41

    def test_erase_on_empty_is_noop_and_no_version_bump(self):
        ml = MaskLayer(100, 120)
        v = ml.version
        assert ml.fill_polygon([(10, 10), (50, 10), (30, 60)], erase=True) is None and ml.version == v

    def test_degenerate_and_outside_are_noop(self):
        ml = MaskLayer(100, 120)
        v = ml.version
        assert ml.fill_polygon([(5, 5), (30, 30), (60, 60)]) is None
        assert ml.fill_polygon([(500, 500), (600, 500), (550, 600)]) is None
        assert ml.version == v and ml.empty

    def test_too_few_points_raises_and_does_not_change(self):
        ml = MaskLayer(100, 120)
        with pytest.raises(InvalidInputError):
            ml.fill_polygon([(1, 1), (2, 2)])
        assert ml.empty and ml.version == 0

    def test_brush_px_untouched_by_fill_but_reset_when_erase_empties(self):
        ml = MaskLayer(100, 120)
        assert ml.fill_polygon([(10, 10), (50, 10), (30, 60)]) is not None and ml.brush_px is None
        ml.clear()
        ml.paint([(5, 5), (8, 8)], 9)
        ml.fill_polygon([(30, 30), (70, 30), (50, 80)])
        assert ml.brush_px == 9
        ml.fill_polygon([(0, 0), (119, 0), (119, 99), (0, 99)], erase=True)
        assert ml.empty and ml.brush_px is None

    def test_clipped_fill_touches_only_image(self):
        ml = MaskLayer(50, 80)
        rect = ml.fill_polygon([(-20, -20), (40, -20), (40, 30), (-20, 30)])
        assert rect == (0, 0, 41, 31) and ml.pixel_count == 41 * 31

    def test_snapshot_restore_roundtrip_for_undo(self):
        ml = MaskLayer(100, 120)
        ml.paint([(10, 10), (60, 60)], 8)
        snap = ml.snapshot()
        ml.fill_polygon([(70, 10), (110, 10), (90, 50)])
        assert not np.array_equal(ml.array, snap)
        ml.restore(snap)
        assert np.array_equal(ml.array, snap)

    def test_lasso_then_components_and_grow(self):
        ml = MaskLayer(100, 120)
        ml.fill_polygon([(10, 10), (30, 10), (30, 30), (10, 30)])
        ml.fill_polygon([(60, 60), (90, 60), (75, 90)])
        comps = ml.components()
        assert len(comps) == 2
        assert ml.grow(3) is not None and ml.pixel_count > sum(c.area for c in comps)


class TestAddWindow:
    def win(self, h, w, val=255):
        return np.full((h, w), val, np.uint8)

    def test_add_union_and_rect(self):
        ml = MaskLayer(100, 120)
        ml.paint([(5, 80), (100, 80)], 6)
        before = ml.snapshot()
        v = ml.version
        w = self.win(20, 30)
        w[:, :10] = 0
        assert ml.add_window((40, 10, 70, 30), w) == (40, 10, 70, 30) and ml.version == v + 1
        a = ml.array
        assert (a[10:30, 50:70] == 255).all() and (a[10:30, 40:50] == 0).all()
        assert (a[before > 0] == 255).all()

    def test_erase_only_selected_pixels(self):
        ml = MaskLayer(50, 60)
        ml.add(np.full((50, 60), 255, np.uint8))
        w = np.zeros((10, 10), np.uint8)
        w[2:5, 3:8] = 7                                 # כל ערך > 0 נחשב נבחר
        assert ml.add_window((10, 10, 20, 20), w, erase=True) is not None
        assert ml.pixel_count == 50 * 60 - 15 and (ml.array[12:15, 13:18] == 0).all()

    def test_noop_cases_do_not_touch(self):
        ml = MaskLayer(50, 60)
        v = ml.version
        assert ml.add_window((0, 0, 10, 10), self.win(10, 10, 0)) is None        # חלון ריק
        assert ml.add_window((0, 0, 10, 10), self.win(10, 10), erase=True) is None   # כבר ריק
        ml.add_window((0, 0, 10, 10), self.win(10, 10))
        v = ml.version
        assert ml.add_window((0, 0, 10, 10), self.win(10, 10)) is None and ml.version == v   # כבר מלא

    @pytest.mark.parametrize("rect", [(-1, 0, 5, 5), (0, 0, 61, 5), (0, 0, 5, 51), (5, 5, 5, 9), (6, 0, 5, 5), "abc", None])
    def test_invalid_rect_raises_and_does_not_change(self, rect):
        ml = MaskLayer(50, 60)
        with pytest.raises(InvalidInputError):
            ml.add_window(rect, self.win(5, 5))
        assert ml.empty and ml.version == 0

    @pytest.mark.parametrize("win", [np.zeros((4, 5), np.uint8), np.zeros((5, 5, 1), np.uint8), [[255] * 5] * 5, None])
    def test_invalid_window_raises(self, win):
        ml = MaskLayer(50, 60)
        with pytest.raises(InvalidInputError):
            ml.add_window((0, 0, 5, 5), win)
        assert ml.empty and ml.version == 0

    def test_brush_px_rules(self):
        ml = MaskLayer(50, 60)
        ml.paint([(30, 30)], 8)
        ml.add_window((0, 0, 10, 10), self.win(10, 10))
        assert ml.brush_px == 8
        ml.add_window((0, 0, 60, 50), self.win(50, 60), erase=True)
        assert ml.empty and ml.brush_px is None

    def test_fill_polygon_equals_add_window_of_region(self):
        pts = star(60, 50, 40, 20, 6, 2)
        a, b = MaskLayer(100, 120), MaskLayer(100, 120)
        rect = a.fill_polygon(pts)
        reg, r2 = mask_ops.polygon_region(pts, (100, 120))
        assert b.add_window(r2, reg) == rect and np.array_equal(a.array, b.array)

    def test_undo_roundtrip_with_history(self):
        from fixpic.session import MaskHistory
        ml = MaskLayer(100, 120)
        h = MaskHistory(ml)
        snap = ml.snapshot()
        h.do(lambda m: m.add_window((10, 10, 40, 40), self.win(30, 30)))
        assert h.undo() == (10, 10, 40, 40) and np.array_equal(ml.array, snap)
        assert h.redo() is not None and ml.pixel_count == 900

    def test_large_image_small_window_is_cheap(self):
        ml = MaskLayer(4000, 6000)
        t0 = time.perf_counter()
        assert ml.add_window((100, 100, 300, 300), self.win(200, 200)) is not None
        assert time.perf_counter() - t0 < 0.3 and ml.pixel_count == 40000
