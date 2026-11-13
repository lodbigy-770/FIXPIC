"""M4 (שירותי מסכה ללא UI): הרחבה/כיווץ, רכיבים מחוברים, Feathering מתכוונן לפי עבודה, ושילוב עם Session."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from fixpic.errors import InvalidInputError
from fixpic.imgio import image_from_array
from fixpic.engine import JobScheduler
from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.pipeline.pipeline import MAX_FEATHER
from fixpic.session import Component, MaskLayer, Session, mask_ops
from test_engine import FILL, FakeModel, rnd

T = 10.0


def blobs(h=200, w=240, seed=0, n=8):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        cv2.circle(m, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 25)), 255, -1)
    return m


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ------------------------------------------------------------------ הרחבה/כיווץ
class TestMorph:
    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("r", [1, 3, 9])
    def test_matches_full_image_cv2(self, seed, r):
        m = blobs(seed=seed)
        g, _ = mask_ops.morph(m, r, grow=True)
        s, _ = mask_ops.morph(m, r, grow=False)
        assert np.array_equal(g, cv2.dilate((m > 0).astype(np.uint8), ell(r)) * 255)
        assert np.array_equal(s, cv2.erode((m > 0).astype(np.uint8), ell(r)) * 255)

    def test_order_properties(self):
        m = blobs(seed=7)
        b = m > 0
        g, _ = mask_ops.morph(m, 4, grow=True)
        s, _ = mask_ops.morph(m, 4, grow=False)
        assert (g > 0)[b].all() and (b | (s > 0)).sum() == b.sum()          # grow ⊇ m ⊇ shrink
        closed, _ = mask_ops.morph(g, 4, grow=False)
        assert (closed > 0)[b].all()                                         # סגירה ⊇ מקור

    def test_input_not_modified_and_values_are_0_255(self):
        m = blobs()
        m[m > 0] = 7                                                         # ערך שרירותי שאינו 255
        orig = m.copy()
        out, _ = mask_ops.morph(m, 3, grow=True)
        assert np.array_equal(m, orig) and set(np.unique(out)) <= {0, 255} and out.dtype == np.uint8

    def test_rect_is_bbox_of_changed_pixels(self):
        m = np.zeros((100, 100), np.uint8)
        m[40:50, 30:60] = 255
        out, rect = mask_ops.morph(m, 5, grow=True)
        ys, xs = np.where(out != m)
        assert rect == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    def test_grow_clips_at_image_edge(self):
        m = np.zeros((50, 50), np.uint8)
        m[0:5, 0:5] = 255
        out, rect = mask_ops.morph(m, 6, grow=True)
        assert out.shape == m.shape and rect[:2] == (0, 0)                  # לא חורג מהתמונה ולא "עוטף"
        assert out[0, 10] == 255 and out[0, 11] == 0 and out[49, 49] == 0

    def test_shrink_does_not_eat_from_image_edge(self):
        m = np.zeros((60, 60), np.uint8)
        m[:, :30] = 255                                                       # נוגעת בשלושה קצוות
        out, _ = mask_ops.morph(m, 4, grow=False)
        assert out[0, 0] == 255 and out[30, 0] == 255 and out[59, 0] == 255   # הקצה נשאר
        assert out[30, 28] == 0 and out[30, 25] == 255                        # הקצה הפנימי התכווץ

    def test_thin_line_vanishes_on_shrink_and_layer_resets_brush(self):
        m = MaskLayer(100, 100)
        m.paint([(10, 50), (90, 50)], 2)
        assert m.brush_px == 2.0
        assert m.shrink(2) is not None and m.empty and m.brush_px is None

    def test_empty_mask_is_noop(self):
        out, rect = mask_ops.morph(np.zeros((20, 20), np.uint8), 3, grow=True)
        assert rect is None and not out.any()

    def test_no_change_returns_none(self):
        m = np.full((30, 30), 255, np.uint8)
        assert mask_ops.morph(m, 3, grow=True)[1] is None                    # כבר מלאה
        assert mask_ops.morph(m, 3, grow=False)[1] is None                   # הקצה לא נאכל

    @pytest.mark.parametrize("px", [0, -1, 1.5, True, "3", None, mask_ops.MAX_MORPH_PX + 1])
    def test_invalid_radius(self, px):
        with pytest.raises(InvalidInputError):
            mask_ops.morph(np.ones((10, 10), np.uint8), px, grow=True)

    @pytest.mark.parametrize("bad", [None, [[1]], np.zeros((5,), np.uint8), np.zeros((0, 5), np.uint8),
                                     np.zeros((4, 4, 3), np.uint8)])
    def test_invalid_mask(self, bad):
        with pytest.raises(InvalidInputError):
            mask_ops.morph(bad, 2, grow=True)

    def test_small_mask_on_24mp_is_cheap(self):
        m = np.zeros((4000, 6000), np.uint8)
        m[2000:2060, 3000:3200] = 255
        t = time.perf_counter()
        out, rect = mask_ops.morph(m, 20, grow=True)
        assert time.perf_counter() - t < 2.0
        assert rect == (2980, 1980, 3220, 2080) and int((out > 0).sum()) > int((m > 0).sum())


# ------------------------------------------------------------------ רכיבים מחוברים
class TestComponents:
    def two(self):
        m = np.zeros((100, 120), np.uint8)
        m[60:70, 10:30] = 255          # שורה 60
        m[10:15, 90:100] = 255         # שורה 10 (מעל)
        return m

    def test_sorted_by_position_with_stable_ids_and_stats(self):
        c = mask_ops.components(self.two())
        assert [x.id for x in c] == [1, 2]
        assert c[0].bbox == (90, 10, 100, 15) and c[0].area == 50
        assert c[1].bbox == (10, 60, 30, 70) and c[1].area == 200
        assert c[1].centroid == pytest.approx((19.5, 64.5)) and isinstance(c[0], Component)

    def test_connectivity_diagonal(self):
        m = np.zeros((10, 10), np.uint8)
        m[2, 2] = m[3, 3] = 255
        assert len(mask_ops.components(m, connectivity=8)) == 1
        assert len(mask_ops.components(m, connectivity=4)) == 2

    def test_min_area_filters_and_empty(self):
        assert len(mask_ops.components(self.two(), min_area=100)) == 1
        assert mask_ops.components(np.zeros((5, 5), np.uint8)) == []

    @pytest.mark.parametrize("kw", [dict(connectivity=6), dict(min_area=0), dict(min_area=1.5), dict(min_area=True)])
    def test_invalid_args(self, kw):
        with pytest.raises(InvalidInputError):
            mask_ops.components(self.two(), **kw)

    def test_component_at(self):
        m = self.two()
        c = mask_ops.component_at(m, 15, 65)
        assert c.shape == m.shape and c.dtype == np.uint8 and set(np.unique(c)) == {0, 255}
        assert np.array_equal(c > 0, np.pad(np.ones((10, 20), bool), ((60, 30), (10, 90))))
        assert mask_ops.component_at(m, 0, 0) is None                        # רקע
        assert mask_ops.component_at(m, -5, 3) is None and mask_ops.component_at(m, 500, 3) is None
        assert (mask_ops.component_at(m, 14.6, 65.4) > 0).sum() == 200       # עיגול קואורדינטות
        assert m[65, 15] == 255 and (m > 0).sum() == 250                      # הקלט לא השתנה

    @pytest.mark.parametrize("pt", [(float("nan"), 1), (1, float("inf")), ("a", 1), (None, 1)])
    def test_component_at_invalid_point(self, pt):
        with pytest.raises(InvalidInputError):
            mask_ops.component_at(self.two(), *pt)

    def test_remove_small(self):
        out, n = mask_ops.remove_small(self.two(), 100)
        assert n == 1 and (out > 0).sum() == 200
        same, n0 = mask_ops.remove_small(self.two(), 10)
        assert n0 == 0 and (same > 0).sum() == 250

    def test_matches_pipeline_grouping_components(self):
        from fixpic.pipeline.grouping import group_mask
        m = blobs(h=400, w=500, seed=3, n=12)
        # קיבוץ ה-Pipeline מאחד רכיבים קרובים, אבל כל רכיב נכלל בקבוצה אחת בדיוק
        comps = mask_ops.components(m)
        groups = group_mask((m > 0).astype(np.uint8), 256)
        assert sum(int(g.mask.sum()) for g in groups) == sum(c.area for c in comps)

    def test_many_components_on_24mp_is_fast(self):
        m = np.zeros((4000, 6000), np.uint8)
        m[100:3900:200, 100:5900:200] = 255
        t = time.perf_counter()
        n = len(mask_ops.components(m))
        assert n == 19 * 29 and time.perf_counter() - t < 3.0
        t = time.perf_counter()
        assert mask_ops.component_at(m, 100, 100) is not None
        assert time.perf_counter() - t < 3.0


# ------------------------------------------------------------------ MaskLayer
class TestMaskLayerEditing:
    def layer(self):
        m = MaskLayer(120, 160)
        m.paint([(30, 30), (60, 30)], 10)
        m.paint([(120, 90)], 20)
        return m

    def test_grow_shrink_bump_version_only_on_change(self):
        m = self.layer()
        v = m.version
        assert m.grow(3) is not None and m.version == v + 1
        before = m.snapshot()
        assert m.shrink(3) is not None and m.version == v + 2 and (m.array > 0)[before == 0].sum() < (before > 0).sum()
        e = MaskLayer(10, 10)
        assert e.grow(2) is None and e.shrink(2) is None and e.version == 0

    def test_grow_then_shrink_never_loses_original(self):
        m = self.layer()
        orig = m.snapshot() > 0
        m.grow(5)
        m.shrink(5)
        assert (m.array > 0)[orig].all()

    def test_components_and_remove_component_at(self):
        m = self.layer()
        assert len(m.components()) == 2
        v = m.version
        assert m.components() and m.component_at(30, 30) is not None and m.version == v     # קריאה בלבד
        rect = m.remove_component_at(120, 90)
        assert rect is not None and len(m.components()) == 1 and m.version == v + 1
        assert m.remove_component_at(120, 90) is None and m.version == v + 1                 # כבר נמחק
        assert m.brush_px == 20.0 or m.brush_px == 10.0                                      # לא ריקה: לא מאופס
        m.remove_component_at(30, 30)
        assert m.empty and m.brush_px is None

    def test_remove_small(self):
        m = self.layer()
        m.paint([(5, 100)], 2)
        assert m.remove_small(30) == 1 and len(m.components()) == 2
        v = m.version
        assert m.remove_small(30) == 0 and m.version == v

    def test_restore_is_undo_for_mask_edits(self):
        m = self.layer()
        snap = m.snapshot()
        m.grow(6)
        m.remove_component_at(30, 30)
        m.restore(snap)
        assert np.array_equal(m.snapshot(), snap) and m.pixel_count == int((snap > 0).sum())
        with pytest.raises(InvalidInputError):
            m.restore(np.zeros((3, 3), np.uint8))
        snap[0, 0] = 99                                                                    # הדבקה לא משאירה הפניה
        assert m.snapshot()[0, 0] == 0

    def test_edits_are_independent_of_earlier_snapshot_and_view(self):
        m = self.layer()
        snap = m.snapshot()
        m.grow(4)
        assert (snap > 0).sum() < m.pixel_count
        with pytest.raises(ValueError):
            m.array[0, 0] = 1


# ------------------------------------------------------------------ Feathering מתכוון
class TestFeatherPerJob:
    def run(self, f, cfg=None):
        img = rnd(200, 200)
        m = np.zeros((200, 200), np.uint8)
        m[60:140, 60:140] = 255
        r = Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=0, **(cfg or {}))).run(img, m, feather_px=f)
        return img, r

    def test_zero_is_binary_and_positive_blends_edge_only(self):
        img, r0 = self.run(0)
        eff = r0.effective_mask.astype(bool)
        assert (r0.image[eff] == FILL).all() and r0.feather_px == 0
        _, r8 = self.run(8)
        e8 = r8.effective_mask.astype(bool)
        assert r8.feather_px == 8 and not (r8.image[e8] == FILL).all()               # קצה מעומעם
        assert (r8.image[70:130, 70:130] == FILL).all()                              # הפנים מלא
        for r in (r0, r8):                                                           # מחוץ למסכה זהה לבייט
            assert np.array_equal(r.image[~r.effective_mask.astype(bool)], img[~r.effective_mask.astype(bool)])

    def test_wider_feather_changes_more_edge_pixels(self):
        _, r4 = self.run(4)
        _, r16 = self.run(16)
        n4 = int((r4.image != FILL).any(-1)[r4.effective_mask.astype(bool)].sum())
        n16 = int((r16.image != FILL).any(-1)[r16.effective_mask.astype(bool)].sum())
        assert n16 > n4 > 0

    def test_none_uses_cfg_and_override_wins(self):
        _, r = self.run(None, dict(feather_px=5))
        assert r.feather_px == 5
        _, r = self.run(0, dict(feather_px=5))
        assert r.feather_px == 0

    @pytest.mark.parametrize("f", [-1, MAX_FEATHER + 1, 2.5, True, "3", float("nan")])
    def test_invalid_feather(self, f):
        with pytest.raises(InvalidInputError):
            self.run(f)

    def test_max_feather_is_accepted(self):
        _, r = self.run(MAX_FEATHER)
        assert r.feather_px == MAX_FEATHER

    def test_flows_through_scheduler_and_session(self):
        sched = JobScheduler(Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=0))).start(warmup=False)
        try:
            s = Session(image_from_array(np.full((120, 120, 3), 50, np.uint8)), sched)
            s.mask.paint([(60, 60)], 40)
            job = s.submit_stroke(size=256, feather_px=0)
            assert job.feather_px == 0 and job.wait(T)
            assert s.finish(job).kind.value == "committed"
            s.mask.paint([(60, 60)], 40)
            job = s.submit_stroke(size=256)
            assert job.feather_px is None and job.wait(T)
            s.finish(job)
            s.mask.paint([(60, 60)], 40)
            job = s.submit_stroke(size=256, feather_px=999)                      # לא תקין: נכשל בעבודה, התמונה לא משתנה
            assert job.wait(T)
            before = s.rgb.copy()
            out = s.finish(job)
            assert out.kind.value == "failed" and isinstance(out.error, InvalidInputError)
            assert np.array_equal(s.rgb, before) and not s.mask.empty              # המסכה הוחזרה לשכבה
        finally:
            sched.shutdown()
