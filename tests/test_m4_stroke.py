"""M4 / AUTO-SELECT שלב 4 (ללא UI): `stroke_to_points` ו-`with_stroke` – משיכה → `Prompts.points` בדגימה אחידה."""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest

from fixpic.select import GrabCutProvider, Point, Prompts, SelectionError, stroke_to_points, with_stroke


def arr(pts):
    return np.array([(p.x, p.y) for p in pts], float)


def dist_to_set(q, pts):
    return np.sqrt(((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(axis=1)


def seg_dist(q, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = np.clip(((q - a) @ d) / max(1e-12, d @ d), 0, 1)
    return np.linalg.norm(q - (a + t[:, None] * d), axis=1)


class TestShape:
    def test_single_click_fills_brush_disk(self):
        pts = stroke_to_points([(100.0, 80.0)], brush_radius=10)
        a = arr(pts)
        assert len(pts) > 20 and all(p.positive for p in pts)
        assert (np.hypot(a[:, 0] - 100, a[:, 1] - 80) <= 10 + 3.0).all()          # לא חורג מהמברשת (עד חצי אלכסון תא)
        q = np.array([(100 + 10 * math.cos(t), 80 + 10 * math.sin(t)) for t in np.linspace(0, 6.28, 40)] + [(100, 80)])
        assert dist_to_set(q, a).max() <= 3.0 * 0.75 + 1e-6                        # כל הדיסק מכוסה בצפיפות

    def test_tiny_brush_still_returns_a_point(self):
        pts = stroke_to_points([(50.3, 20.7)], brush_radius=0.2)
        assert len(pts) == 1

    def test_thin_diagonal_stroke_is_covered_without_gaps(self):
        a, b = (20.0, 20.0), (220.0, 140.0)
        pts = arr(stroke_to_points([a, b], brush_radius=1.0))
        t = np.linspace(0, 1, 400)[:, None]
        q = np.array(a) * (1 - t) + np.array(b) * t
        assert dist_to_set(q, pts).max() <= 3.0 + 1e-6
        assert (seg_dist(pts, a, b) <= 1.0 + 3.0).all()

    def test_wide_stroke_uniform_area_coverage(self):
        path = [(40.0, 60.0), (160.0, 60.0), (160.0, 140.0)]
        pts = arr(stroke_to_points(path, brush_radius=14))
        rng = np.random.default_rng(0)
        q = rng.uniform((20, 40), (190, 170), (3000, 2))
        inside = np.minimum(seg_dist(q, path[0], path[1]), seg_dist(q, path[1], path[2])) <= 14
        assert dist_to_set(q[inside], pts).max() <= 3.0 * 0.75 + 1e-6              # כל השטח שהמברשת כיסתה
        d_out = np.minimum(seg_dist(pts, path[0], path[1]), seg_dist(pts, path[1], path[2]))
        assert (d_out <= 14 + 2.2).all()                                          # ושום דבר מחוץ לו (עד חצי אלכסון תא)

    def test_negative_flag(self):
        pts = stroke_to_points([(10.0, 10.0), (50.0, 10.0)], brush_radius=4, positive=False)
        assert pts and not any(p.positive for p in pts)


class TestUniformity:
    def test_independent_of_event_rate(self):
        """אותו קו, פעם בשני אירועים ופעם ב-200 – אותה תוצאה (הדגימה לפי שטח ולא לפי אירועים)."""
        sparse = [(10.0, 30.0), (210.0, 30.0)]
        dense = [(10.0 + i, 30.0) for i in range(201)]
        assert stroke_to_points(sparse, brush_radius=5) == stroke_to_points(dense, brush_radius=5)

    def test_duplicate_events_and_retrace_do_not_duplicate_points(self):
        base = [(10.0, 30.0), (110.0, 30.0)]
        a = stroke_to_points(base, brush_radius=5)
        b = stroke_to_points([base[0], base[0], base[1], base[1], base[0], base[1]], brush_radius=5)
        assert a == b and len({(p.x, p.y) for p in a}) == len(a)

    def test_deterministic(self):
        path = [(5.5, 7.5), (80.2, 33.3), (120.0, 10.0), (60.0, 90.0)]
        assert stroke_to_points(path, brush_radius=9) == stroke_to_points(list(path), brush_radius=9)

    def test_spacing_controls_density(self):
        p = [(10.0, 10.0), (300.0, 10.0)]
        coarse = stroke_to_points(p, brush_radius=3, spacing=9.0)
        fine = stroke_to_points(p, brush_radius=3, spacing=3.0)
        assert 0 < len(coarse) < len(fine)

    def test_max_points_cap_stays_uniform_over_whole_path(self):
        path = [(10.0, 50.0), (900.0, 50.0), (900.0, 400.0)]
        pts = stroke_to_points(path, brush_radius=30, max_points=300)
        a = arr(pts)
        assert 100 < len(pts) <= 300
        assert a[:, 0].min() < 30 and a[:, 1].max() > 380 and a[:, 0].max() > 900 - 5       # גם הסוף מכוסה
        q = np.array([(10.0 + i * 8.9, 50.0) for i in range(100)] + [(900.0, 50.0 + i * 3.5) for i in range(100)])
        assert dist_to_set(q, a).max() < 60                                        # בלי חורים גדולים

    def test_huge_stroke_is_bounded(self):
        t0 = time.perf_counter()
        pts = stroke_to_points([(0.0, 0.0), (5500.0, 3900.0)], brush_radius=400, bounds=(6000, 4000))
        assert 0 < len(pts) <= 2000 and time.perf_counter() - t0 < 1.0


class TestBounds:
    def test_points_clipped_to_image(self):
        pts = arr(stroke_to_points([(-30.0, 50.0), (60.0, 50.0)], brush_radius=10, bounds=(100, 80)))
        assert len(pts) and (pts[:, 0] >= 0).all() and (pts[:, 0] < 100).all() and (pts[:, 1] >= 0).all()
        pts = arr(stroke_to_points([(50.0, 50.0), (200.0, 79.0)], brush_radius=15, bounds=(100, 80)))
        assert (pts[:, 0] < 100).all() and (pts[:, 1] < 80).all()

    def test_entirely_outside_gives_empty(self):
        assert stroke_to_points([(-50.0, -50.0), (-20.0, -20.0)], brush_radius=5, bounds=(100, 100)) == ()
        assert stroke_to_points([(500.0, 5.0)], brush_radius=5, bounds=(100, 100)) == ()

    def test_tiny_click_at_edge_inside(self):
        pts = stroke_to_points([(0.0, 0.0)], brush_radius=0.1, bounds=(10, 10))
        assert pts == (Point(0.0, 0.0, True),)

    def test_no_bounds_allows_any_coordinates(self):
        assert stroke_to_points([(-5.0, -5.0)], brush_radius=2)


class TestValidation:
    @pytest.mark.parametrize("path", [[], (), None, [(1.0,)], [(1, 2, 3)], [("a", "b")], [(float("nan"), 1.0)],
                                      [(1.0, float("inf"))], 5])
    def test_bad_path(self, path):
        with pytest.raises(SelectionError):
            stroke_to_points(path, brush_radius=5)

    @pytest.mark.parametrize("r", [0, -1, float("nan"), float("inf"), "x", None, True])
    def test_bad_radius(self, r):
        with pytest.raises(SelectionError):
            stroke_to_points([(5.0, 5.0)], brush_radius=r)

    @pytest.mark.parametrize("kw", [{"spacing": 0}, {"spacing": -2}, {"spacing": float("nan")}, {"max_points": 0},
                                    {"max_points": 2.5}, {"max_points": True}, {"bounds": (0, 10)}, {"bounds": (10,)},
                                    {"bounds": "ab"}])
    def test_bad_options(self, kw):
        with pytest.raises(SelectionError):
            stroke_to_points([(5.0, 5.0)], brush_radius=3, **kw)

    def test_too_long_path(self):
        with pytest.raises(SelectionError):
            stroke_to_points(np.zeros((200_001, 2)) + np.arange(200_001)[:, None], brush_radius=1)

    def test_accepts_numpy_and_ints(self):
        assert stroke_to_points(np.array([[10, 10], [40, 10]]), brush_radius=np.float32(4), max_points=np.int64(500))


class TestWithStroke:
    def test_appends_and_keeps_other_fields(self):
        p0 = Prompts(box=(1, 2, 90, 80), points=(Point(5, 5, True),), tolerance=33)
        p1 = with_stroke(p0, [(30.0, 30.0), (60.0, 30.0)], brush_radius=4, positive=False)
        assert p0.points == (Point(5, 5, True),)                                  # המקור לא השתנה
        assert p1.points[0] == Point(5, 5, True) and len(p1.points) > 5
        assert not any(p.positive for p in p1.points[1:])
        assert (p1.box, p1.tolerance, p1.polygon) == (p0.box, 33, None)

    def test_two_strokes_accumulate(self):
        p = with_stroke(Prompts(), [(10.0, 10.0), (50.0, 10.0)], brush_radius=3)
        p = with_stroke(p, [(10.0, 60.0), (50.0, 60.0)], brush_radius=3, positive=False)
        assert p.positives and p.negatives and len(p.points) == len(p.positives) + len(p.negatives)

    def test_frozen_prompts_are_replaced_not_mutated(self):
        p0 = Prompts()
        p1 = with_stroke(p0, [(1.0, 1.0)], brush_radius=2)
        assert p1 is not p0 and dataclasses.is_dataclass(p1) and p0.points == ()


class TestWithGrabCut:
    def test_stroke_removes_attached_region_where_a_single_click_does_not(self):
        """אותה סצנה כמו ב-`test_negative_points_remove_attached_region`, אבל מאירועי עכבר גולמיים (לא אחידים, עם כפילויות)."""
        rng = np.random.default_rng(7)
        img = np.full((300, 400, 3), (60, 140, 60), np.float32)
        img[80:220, 80:220] = (200, 60, 60)
        img[120:160, 220:360] = (190, 70, 70)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        gt = np.zeros((300, 400), bool)
        gt[80:220, 80:220] = True
        prov = GrabCutProvider()
        ctx = prov.prepare(img)
        box = (60, 60, 380, 240)

        def region(prompts):
            prop = prov.propose(ctx, prompts)
            m = np.zeros(gt.shape, bool)
            x0, y0, x1, y1 = prop.rect
            m[y0:y1, x0:x1] = prop.window > 0
            return m

        events = [(235.0, 139.0), (235.0, 139.0), (251.5, 141.0), (299.0, 140.0), (300.0, 140.0), (341.0, 138.5), (358.0, 140.0)]
        single = region(Prompts(box=box, points=(Point(300, 140, False),)))
        stroke = region(with_stroke(Prompts(box=box), events, brush_radius=8, positive=False, bounds=(400, 300)))
        assert single[120:160, 250:360].mean() > 0.5                               # לחיצה בודדת לא מספיקה
        assert (stroke & gt).sum() / (stroke | gt).sum() > 0.93 and stroke[120:160, 250:360].mean() < 0.1
