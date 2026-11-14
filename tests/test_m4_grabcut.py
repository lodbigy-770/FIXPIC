"""M4 / AUTO-SELECT שלב 3: `GrabCutProvider` (מלבן/לאסו/נקודות על ROI מוקטן). תמונות סינתטיות עם רעש; האיכות על תמונות
אמיתיות נמדדת ב-`tools/m4/grabcut_samples.py` (בדיקה בעין), לא כאן."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fixpic.errors import CancelledError, InvalidInputError
from fixpic.models.base import CancelToken
from fixpic.select import (GrabCutProvider, Point, Prompts, SelectionError, SelectionProvider, SelectionService)
from fixpic.select.grabcut import KEEP_RATIO
from fixpic.session import MaskHistory, MaskLayer


def scene(h=240, w=320, seed=1):
    """רקע ירוק-אפור עם רעש, ושני אובייקטים: אליפסה אדומה (מרכז 110,100) וריבוע כחול (מרכז 240,170)."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (90, 120, 80)
    yy, xx = np.mgrid[:h, :w]
    ell = ((xx - 110) / 45.0) ** 2 + ((yy - 100) / 35.0) ** 2 <= 1
    sq = (abs(xx - 240) <= 30) & (abs(yy - 170) <= 30)
    img[ell] = (200, 40, 40)
    img[sq] = (40, 50, 200)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), ell, sq


def full(prop, shape):
    m = np.zeros(shape, bool)
    if prop is not None:
        x0, y0, x1, y1 = prop.rect
        m[y0:y1, x0:x1] = prop.window > 0
    return m


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.fixture(scope="module")
def sc():
    return scene()


@pytest.fixture()
def prov():
    return GrabCutProvider()


# ------------------------------------------------------------------------------------------ חוזה
class TestContract:
    def test_protocol_and_flags(self, prov):
        assert isinstance(prov, SelectionProvider)
        assert prov.name == "grabcut" and prov.heavy is True and prov.stale_ok is False

    def test_constructor_validation(self):
        for kw in ({"iterations": 0}, {"iterations": 21}, {"max_roi_px": 10}, {"max_analysis_px": 10},
                   {"margin": -0.1}, {"margin": 1.5}, {"hint_frac": -1}, {"hint_frac": 0.9}):
            with pytest.raises(ValueError):
                GrabCutProvider(**kw)

    def test_prepare_validation_and_private_copy(self, prov, sc):
        img = sc[0]
        for bad in (img.astype(np.float32), img[:, :, 0], img[:, :, :2], np.zeros((0, 5, 3), np.uint8), [1, 2]):
            with pytest.raises(InvalidInputError):
                prov.prepare(bad)
        ctx = prov.prepare(img)
        assert ctx.provider == "grabcut" and ctx.shape == img.shape[:2] and ctx.nbytes == img.nbytes
        assert not np.shares_memory(ctx.data, img)

    def test_prepare_downscales_large_image(self):
        p = GrabCutProvider(max_analysis_px=40_000)
        ctx = p.prepare(np.zeros((400, 600, 3), np.uint8))
        assert ctx.shape == (400, 600) and ctx.data.shape[0] * ctx.data.shape[1] <= 40_500

    def test_prepare_cancel(self, prov, sc):
        tok = CancelToken()
        tok.cancel()
        with pytest.raises(CancelledError):
            prov.prepare(sc[0], tok)

    def test_ctx_of_other_provider(self, prov, sc):
        ctx = prov.prepare(sc[0])
        from fixpic.select import Context
        with pytest.raises(SelectionError):
            prov.propose(Context("magic-wand", ctx.shape, ctx.data), Prompts(box=(50, 50, 150, 150)))


# ------------------------------------------------------------------------------------------ מלבן
class TestBox:
    def test_box_selects_ellipse(self, prov, sc):
        img, ell, _ = sc
        ctx = prov.prepare(img)
        prop = prov.propose(ctx, Prompts(box=(55, 55, 165, 145)))
        assert prop is not None and iou(full(prop, ell.shape), ell) > 0.95
        x0, y0, x1, y1 = prop.rect
        assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240 and prop.window.shape == (y1 - y0, x1 - x0)
        assert prop.window.dtype == np.uint8 and set(np.unique(prop.window)) <= {0, 255} and prop.alternatives == ()

    def test_loose_box_still_works(self, prov, sc):
        img, ell, _ = sc
        prop = prov.propose(prov.prepare(img), Prompts(box=(20, 20, 200, 170)))
        assert iou(full(prop, ell.shape), ell) > 0.9

    def test_deterministic(self, prov, sc):
        ctx = prov.prepare(sc[0])
        a = prov.propose(ctx, Prompts(box=(55, 55, 165, 145)))
        b = prov.propose(ctx, Prompts(box=(55, 55, 165, 145)))
        assert a.rect == b.rect and np.array_equal(a.window, b.window)

    def test_does_not_modify_inputs(self, prov, sc):
        img = sc[0].copy()
        ctx = prov.prepare(img)
        before = ctx.data.copy()
        prov.propose(ctx, Prompts(box=(55, 55, 165, 145)))
        assert np.array_equal(img, sc[0]) and np.array_equal(ctx.data, before)

    def test_box_clipped_to_image(self, prov, sc):
        img, ell, _ = sc
        prop = prov.propose(prov.prepare(img), Prompts(box=(-50, -30, 170, 150)))
        assert prop is not None and iou(full(prop, ell.shape), ell) > 0.9

    def test_object_at_image_corner(self, prov):
        img = np.full((200, 200, 3), (60, 140, 60), np.uint8)
        img[0:60, 0:70] = (210, 60, 60)
        img = np.clip(img + np.random.default_rng(0).normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        prop = prov.propose(prov.prepare(img), Prompts(box=(0, 0, 90, 80)))
        gt = np.zeros((200, 200), bool)
        gt[0:60, 0:70] = True
        assert iou(full(prop, gt.shape), gt) > 0.95 and prop.rect[:2] == (0, 0)

    def test_two_objects_in_box_both_kept(self, prov, sc):
        img, ell, sq = sc
        prop = prov.propose(prov.prepare(img), Prompts(box=(55, 55, 285, 215)))
        m = full(prop, ell.shape)
        assert iou(m, ell | sq) > 0.9 and (m & ell).sum() > 0.9 * ell.sum() and (m & sq).sum() > 0.9 * sq.sum()

    def test_small_speck_is_dropped(self, prov):
        img = np.full((260, 320, 3), (70, 130, 70), np.uint8)
        img[60:160, 60:200] = (200, 50, 50)        # אובייקט
        img[200:204, 250:254] = (200, 50, 50)      # כתם קטן (16 פיקסלים, הרבה מתחת ל-20%)
        img = np.clip(img + np.random.default_rng(3).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        prop = prov.propose(prov.prepare(img), Prompts(box=(40, 40, 270, 225)))
        m = full(prop, img.shape[:2])
        assert not m[196:208, 246:258].any() and m[60:160, 60:200].mean() > 0.95
        assert KEEP_RATIO == 0.2

    def test_large_image_roi_downscaled(self):
        """תמונה גדולה: ה-ROI מוקטן לתקציב והמסכה מוחזרת ברזולוציה מלאה."""
        h, w = 1500, 2000
        img = np.full((h, w, 3), (80, 130, 70), np.uint8)
        yy, xx = np.mgrid[:h, :w]
        gt = ((xx - 1000) / 400.0) ** 2 + ((yy - 700) / 300.0) ** 2 <= 1
        img[gt] = (210, 50, 50)
        img = np.clip(img + np.random.default_rng(5).normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        p = GrabCutProvider(max_roi_px=60_000, max_analysis_px=1_000_000)
        ctx = p.prepare(img)
        t = time.perf_counter()
        prop = p.propose(ctx, Prompts(box=(550, 350, 1450, 1050)))
        assert time.perf_counter() - t < 10
        assert iou(full(prop, gt.shape), gt) > 0.95
        assert prop.window.shape == (prop.rect[3] - prop.rect[1], prop.rect[2] - prop.rect[0])


# ------------------------------------------------------------------------------------------ לאסו
class TestLasso:
    def test_polygon_selects_ellipse(self, prov, sc):
        img, ell, _ = sc
        poly = ((50, 55), (170, 55), (170, 148), (50, 148))
        prop = prov.propose(prov.prepare(img), Prompts(polygon=poly))
        assert iou(full(prop, ell.shape), ell) > 0.95

    def test_irregular_polygon_and_clipping(self, prov, sc):
        img, ell, _ = sc
        poly = ((40, 40), (110, 30), (180, 60), (175, 150), (100, 160), (45, 130), (-20, 100))
        prop = prov.propose(prov.prepare(img), Prompts(polygon=poly))
        assert iou(full(prop, ell.shape), ell) > 0.9

    @pytest.mark.parametrize("poly", [((1, 1), (5, 5)), (), ((1, 1), (2, 2), (float("nan"), 3)), "abc", ((1, 2, 3),) * 4])
    def test_bad_polygon(self, prov, sc, poly):
        with pytest.raises(SelectionError):
            prov.propose(prov.prepare(sc[0]), Prompts(polygon=poly))

    def test_too_many_points(self, prov, sc):
        poly = tuple((50 + (i % 50), 50 + (i % 40)) for i in range(30_001))
        with pytest.raises(SelectionError):
            prov.propose(prov.prepare(sc[0]), Prompts(polygon=poly))


# ------------------------------------------------------------------------------------------ נקודות
class TestPoints:
    def test_positive_point_keeps_small_component(self, prov):
        img = np.full((260, 320, 3), (70, 130, 70), np.uint8)
        img[60:160, 60:200] = (200, 50, 50)
        img[200:206, 250:256] = (30, 40, 200)      # כתם קטן וברור, בתוך המלבן
        img = np.clip(img + np.random.default_rng(3).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        ctx = prov.prepare(img)
        box = (40, 40, 270, 225)
        without = full(prov.propose(ctx, Prompts(box=box)), img.shape[:2])
        with_pt = full(prov.propose(ctx, Prompts(box=box, points=(Point(253, 203),))), img.shape[:2])
        assert not without[200:206, 250:256].any() and with_pt[200:206, 250:256].mean() > 0.8

    def test_negative_points_remove_attached_region(self, prov):
        """אובייקט + מבנה דומה לו שנוגע בו ונכלל במלבן; משיכה (נקודות שליליות צפופות) על המבנה
        מוציאה אותו. נקודות בודדות לא מספיקות (נמדד: 3 נקודות רחוקות השאירו 94% מהזנב), ולכן הממשק שולח משיכה."""
        rng = np.random.default_rng(7)
        img = np.full((300, 400, 3), (60, 140, 60), np.float32)
        img[80:220, 80:220] = (200, 60, 60)
        img[120:160, 220:360] = (190, 70, 70)             # זנב דומה בצבעו, מחובר לאובייקט
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        gt = np.zeros((300, 400), bool)
        gt[80:220, 80:220] = True
        ctx = prov.prepare(img)
        box = (60, 60, 380, 240)
        base = full(prov.propose(ctx, Prompts(box=box)), gt.shape)
        negs = tuple(Point(x, 140, False) for x in range(235, 360, 10))
        fixed = full(prov.propose(ctx, Prompts(box=box, points=negs)), gt.shape)
        assert base[120:160, 230:360].mean() > 0.5          # בלי נקודות: הזנב נבחר
        assert iou(fixed, gt) > 0.93 and fixed[120:160, 250:360].mean() < 0.1

    def test_negative_point_outside_roi_is_ignored(self, prov, sc):
        img, ell, _ = sc
        ctx = prov.prepare(img)
        prop = prov.propose(ctx, Prompts(box=(55, 55, 165, 145), points=(Point(300, 220, False),)))
        assert iou(full(prop, ell.shape), ell) > 0.95

    def test_positive_point_outside_roi(self, prov, sc):
        with pytest.raises(SelectionError):
            prov.propose(prov.prepare(sc[0]), Prompts(box=(55, 55, 165, 145), points=(Point(300, 220),)))

    @pytest.mark.parametrize("pt", [Point(-1, 5), Point(5, -1), Point(320, 5), Point(5, 240), Point(float("nan"), 5),
                                    Point(float("inf"), 5)])
    def test_point_outside_image_or_not_finite(self, prov, sc, pt):
        with pytest.raises(SelectionError):
            prov.propose(prov.prepare(sc[0]), Prompts(box=(55, 55, 165, 145), points=(pt,)))


# ------------------------------------------------------------------------------------------ קלט לא תקין
class TestInvalid:
    @pytest.mark.parametrize("prompts", [
        Prompts(),                                                           # בלי מלבן/לאסו
        Prompts(points=(Point(100, 100),)),                                   # נקודות בלבד
        Prompts(box=(50, 50, 150, 150), polygon=((50, 50), (150, 50), (150, 150))),   # שניהם
        Prompts(box=(50, 50, 150, 150), tolerance=10),
        Prompts(box=(50, 50, 52, 52)),                                         # קטן מדי
        Prompts(box=(50, 50, 50, 150)),                                        # ריק
        Prompts(box=(400, 400, 500, 500)),                                     # מחוץ לתמונה
        Prompts(box="abc"),
        Prompts(box=(0, 0, 320, 240)),                                         # כל התמונה: אין רקע
    ])
    def test_selection_error(self, prov, sc, prompts):
        with pytest.raises(SelectionError):
            prov.propose(prov.prepare(sc[0]), prompts)

    def test_tiny_image(self, prov):
        ctx = prov.prepare(np.zeros((6, 6, 3), np.uint8))
        with pytest.raises(SelectionError):
            prov.propose(ctx, Prompts(box=(0, 0, 6, 6)))

    def test_uniform_image_does_not_crash(self, prov):
        """תמונה אחידה: או הצעה או None או SelectionError, אבל לא חריגה אחרת."""
        ctx = prov.prepare(np.full((100, 100, 3), 120, np.uint8))
        try:
            r = prov.propose(ctx, Prompts(box=(30, 30, 70, 70)))
        except SelectionError:
            return
        assert r is None or r.rect[2] <= 100


# ------------------------------------------------------------------------------------------ ביטול
class TestCancel:
    def test_cancel_before_run(self, prov, sc):
        ctx = prov.prepare(sc[0])
        tok = CancelToken()
        tok.cancel()
        with pytest.raises(CancelledError):
            prov.propose(ctx, Prompts(box=(55, 55, 165, 145)), tok)

    def test_cancel_between_iterations(self, sc):
        class Late(CancelToken):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def raise_if_cancelled(self):
                self.calls += 1
                if self.calls >= 3:
                    self.cancel()
                super().raise_if_cancelled()

        p = GrabCutProvider(iterations=8)
        tok = Late()
        with pytest.raises(CancelledError):
            p.propose(p.prepare(sc[0]), Prompts(box=(55, 55, 165, 145)), tok)
        assert tok.calls == 3                      # נעצר אחרי 2 איטרציות, לא אחרי כולן


# ------------------------------------------------------------------------------------------ Service ו-MaskLayer
class TestIntegration:
    def test_through_service_to_mask_layer_one_undo_step(self, sc):
        img, ell, _ = sc
        with SelectionService(GrabCutProvider()) as svc:
            req = svc.propose(img, "k1", Prompts(box=(55, 55, 165, 145)))
            prop = req.result(timeout=30)
        assert prop is not None and iou(full(prop, ell.shape), ell) > 0.95
        layer = MaskLayer(img.shape[0], img.shape[1])
        hist = MaskHistory(layer)
        assert hist.do(lambda m: prop.add_to(m)) == prop.rect
        assert layer.bbox() is not None and hist.undo() and layer.bbox() is None

    def test_heavy_waits_for_busy(self, sc):
        busy = threading.Event()
        busy.set()
        with SelectionService(GrabCutProvider(), busy=busy.is_set, busy_poll=0.01) as svc:
            req = svc.propose(sc[0], "k", Prompts(box=(55, 55, 165, 145)))
            assert not req.wait(0.3)               # Inpaint "פעיל": הבחירה ממתינה
            busy.clear()
            assert req.wait(30) and req.proposal is not None

    def test_service_reports_selection_error(self, sc):
        with SelectionService(GrabCutProvider()) as svc:
            req = svc.propose(sc[0], "k", Prompts())
            req.wait(10)
            assert isinstance(req.error, SelectionError)
