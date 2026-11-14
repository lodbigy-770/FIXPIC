"""M4 / AUTO-SELECT שלב 2: חבילת `fixpic.select` (types, Protocol, MagicWandProvider, SelectionService). ללא UI וללא מודל."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fixpic.errors import CancelledError, FixPicError, SchedulerClosedError
from fixpic.models.base import CancelToken
from fixpic.select import (Context, MagicWandProvider, Point, Prompts, Proposal, RequestState, SelectionError,
                           SelectionProvider, SelectionService)
from fixpic.session import MaskHistory, MaskLayer


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return bool(cond())


def scene(h=200, w=300):
    """רקע אפור, מלבן אדום (50..120 x 40..100), מלבן כחול (180..260 x 120..180)."""
    img = np.full((h, w, 3), 128, np.uint8)
    img[40:100, 50:120] = (220, 20, 20)
    img[120:180, 180:260] = (20, 20, 220)
    return img


def sel(win, rect, shape):
    m = np.zeros(shape, np.uint8)
    m[rect[1]:rect[3], rect[0]:rect[2]] = win
    return m


# ------------------------------------------------------------------------------------------ טיפוסים
class TestTypes:
    def test_proposal_validation(self):
        Proposal(np.full((4, 6), 255, np.uint8), (10, 20, 16, 24))
        for win, rect in [(np.zeros((4, 6), np.float32), (0, 0, 6, 4)), (np.zeros((4, 6), np.uint8), (0, 0, 5, 4)),
                          (np.zeros((6,), np.uint8), (0, 0, 6, 1)), (np.zeros((4, 6), np.uint8), (5, 5, 5, 9))]:
            with pytest.raises(SelectionError):
                Proposal(win, rect)

    def test_prompts_split_points(self):
        p = Prompts(points=(Point(1, 2), Point(3, 4, False), Point(5, 6, True)))
        assert [q.x for q in p.positives] == [1, 5] and [q.x for q in p.negatives] == [3]

    def test_selection_error_is_fixpic_error(self):
        assert issubclass(SelectionError, FixPicError)

    def test_add_to_layer_with_history_is_one_undo_step(self):
        layer = MaskLayer(50, 60)
        hist = MaskHistory(layer)
        prop = Proposal(np.full((10, 20), 255, np.uint8), (5, 5, 25, 15))
        assert hist.do(lambda m: prop.add_to(m)) == (5, 5, 25, 15)
        assert layer.pixel_count == 200 and prop.pixel_count == 200
        assert hist.do(lambda m: prop.add_to(m)) is None        # כבר מלא: לא נרשם
        hist.undo()
        assert layer.empty
        hist.redo()
        hist.do(lambda m: Proposal(np.full((2, 2), 255, np.uint8), (6, 6, 8, 8)).add_to(m, erase=True))
        assert layer.pixel_count == 196


# ------------------------------------------------------------------------------------------ שרביט
class TestWand:
    def prep(self, img=None, **kw):
        p = MagicWandProvider(**kw)
        img = scene() if img is None else img
        return p, img, p.prepare(img)

    def test_protocol_and_flags(self):
        p = MagicWandProvider()
        assert isinstance(p, SelectionProvider) and p.heavy is False and p.stale_ok is False

    def test_click_selects_exact_rectangle(self):
        p, img, ctx = self.prep()
        pr = p.propose(ctx, Prompts(points=(Point(80, 70),)))
        assert pr.rect == (50, 40, 120, 100) and pr.window.all()
        assert pr.window.dtype == np.uint8 and set(np.unique(pr.window)) == {255}

    def test_background_selection_has_holes(self):
        p, img, ctx = self.prep()
        pr = p.propose(ctx, Prompts(points=(Point(5, 5),)))
        m = sel(pr.window, pr.rect, img.shape[:2])
        assert pr.rect == (0, 0, 300, 200)
        assert m[70, 80] == 0 and m[150, 200] == 0 and m[5, 5] == 255
        assert (m > 0).sum() == 300 * 200 - 70 * 60 - 80 * 60

    def test_two_positives_union(self):
        p, img, ctx = self.prep()
        pr = p.propose(ctx, Prompts(points=(Point(80, 70), Point(200, 150))))
        m = sel(pr.window, pr.rect, img.shape[:2])
        assert pr.rect == (50, 40, 260, 180) and (m > 0).sum() == 70 * 60 + 80 * 60

    def test_negative_point_subtracts(self):
        p, img, ctx = self.prep()
        pr = p.propose(ctx, Prompts(points=(Point(5, 5), Point(80, 70, False))))
        m = sel(pr.window, pr.rect, img.shape[:2])
        assert m[70, 80] == 0 and m[150, 200] == 0          # הנקודה השלילית לא בחירה; הכחול לא נבחר גם ככה

    def test_all_negative_gives_none(self):
        p, img, ctx = self.prep()
        assert p.propose(ctx, Prompts(points=(Point(80, 70), Point(80, 70, False)))) is None

    def test_tolerance_gradient(self):
        img = np.zeros((40, 100, 3), np.uint8)
        img[:] = np.arange(100, dtype=np.uint8)[None, :, None] * 2        # 0..198 לאורך
        p, _, ctx = self.prep(img)
        narrow = p.propose(ctx, Prompts(points=(Point(0, 5),), tolerance=10))
        wide = p.propose(ctx, Prompts(points=(Point(0, 5),), tolerance=100))
        assert narrow.rect[2] == 6 and wide.rect[2] == 51                 # |2x| <= tol, לפי ערך הזרע
        assert narrow.pixel_count == 40 * 6 and wide.pixel_count == 40 * 51

    def test_default_tolerance_from_constructor(self):
        img = np.zeros((10, 100, 3), np.uint8)
        img[:] = np.arange(100, dtype=np.uint8)[None, :, None] * 2
        p, _, ctx = self.prep(img, tolerance=10)
        assert p.propose(ctx, Prompts(points=(Point(0, 5),))).rect[2] == 6

    def test_box_clips_search_area(self):
        p, img, ctx = self.prep()
        pr = p.propose(ctx, Prompts(points=(Point(5, 5),), box=(0, 0, 150, 100)))
        assert pr.rect == (0, 0, 150, 100)
        m = sel(pr.window, pr.rect, img.shape[:2])
        assert m[70, 80] == 0 and m[5, 5] == 255
        with pytest.raises(SelectionError):
            p.propose(ctx, Prompts(points=(Point(200, 150),), box=(0, 0, 150, 100)))      # זרע מחוץ ל-box

    def test_invalid_inputs(self):
        p, img, ctx = self.prep()
        for bad in (Prompts(), Prompts(points=(Point(1, 1, False),)), Prompts(points=(Point(300, 5),)),
                    Prompts(points=(Point(-1, 5),)), Prompts(points=(Point(float("nan"), 5),)),
                    Prompts(points=(Point(5, 5),), polygon=((0, 0), (5, 5), (0, 5))),
                    Prompts(points=(Point(5, 5),), tolerance=256), Prompts(points=(Point(5, 5),), tolerance=-1),
                    Prompts(points=(Point(5, 5),), box=(500, 500, 600, 600))):
            with pytest.raises(SelectionError):
                p.propose(ctx, bad)
        with pytest.raises(SelectionError):
            p.propose(Context("other", (200, 300)), Prompts(points=(Point(5, 5),)))
        with pytest.raises(SelectionError):
            MagicWandProvider(tolerance=300)

    def test_prepare_validates_and_isolates(self):
        p = MagicWandProvider()
        with pytest.raises(FixPicError):
            p.prepare(np.zeros((5, 5), np.uint8))
        img = scene()
        ctx = p.prepare(img)
        img[:] = 0                                                         # שינוי המקור אינו משפיע על ה-Context
        before = ctx.data.copy()
        assert p.propose(ctx, Prompts(points=(Point(80, 70),), box=(0, 0, 200, 150))).rect == (50, 40, 120, 100)
        assert np.array_equal(ctx.data, before) and ctx.nbytes == ctx.data.nbytes     # propose לא משנה את ה-Context

    def test_prepare_cancelled(self):
        tok = CancelToken()
        tok.cancel()
        with pytest.raises(CancelledError):
            MagicWandProvider().prepare(scene(), tok)

    def test_propose_cancelled(self):
        p, img, ctx = self.prep()
        tok = CancelToken()
        tok.cancel()
        with pytest.raises(CancelledError):
            p.propose(ctx, Prompts(points=(Point(80, 70),)), tok)

    def test_large_image_downscaled_but_full_resolution_result(self):
        h, w = 2000, 3000
        img = np.full((h, w, 3), 128, np.uint8)
        img[400:1000, 500:1200] = (220, 20, 20)
        p = MagicWandProvider(max_analysis_px=100_000)
        ctx = p.prepare(img)
        assert ctx.data.shape[0] * ctx.data.shape[1] <= 100_000 and ctx.shape == (h, w)
        pr = p.propose(ctx, Prompts(points=(Point(800, 700),)))
        x0, y0, x1, y1 = pr.rect
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and pr.window.shape == (y1 - y0, x1 - x0)
        truth = np.zeros((h, w), bool)
        truth[400:1000, 500:1200] = True
        got = np.zeros((h, w), bool)
        got[y0:y1, x0:x1] = pr.window > 0
        iou = (truth & got).sum() / (truth | got).sum()
        assert iou > 0.97
        assert abs(x0 - 500) <= 12 and abs(x1 - 1200) <= 12            # שגיאת קצה: עד כ-1 פיקסל ניתוח (הקטנה כ-×5.5)

    def test_corner_clamps_inside_image(self):
        img = scene()
        img[-3:, -5:] = (1, 250, 1)
        p = MagicWandProvider(max_analysis_px=2000)
        pr = p.propose(p.prepare(img), Prompts(points=(Point(299, 199),)))
        assert pr is not None and pr.rect[2] <= 300 and pr.rect[3] <= 200

    def test_tiny_images(self):
        p = MagicWandProvider()
        pr = p.propose(p.prepare(np.zeros((1, 1, 3), np.uint8)), Prompts(points=(Point(0, 0),)))
        assert pr.rect == (0, 0, 1, 1) and pr.window.shape == (1, 1)

    def test_24mp_speed_smoke(self):
        img = np.full((4000, 6000, 3), 90, np.uint8)
        p = MagicWandProvider()
        t0 = time.perf_counter()
        ctx = p.prepare(img)
        t1 = time.perf_counter()
        pr = p.propose(ctx, Prompts(points=(Point(3000, 2000),)))
        t2 = time.perf_counter()
        assert pr.rect == (0, 0, 6000, 4000)
        assert t1 - t0 < 3 and t2 - t1 < 3          # רף רופף לסביבות איטיות; המדידה האמיתית אצל המשתמש


# ------------------------------------------------------------------------------------------ Service
class Fake:
    """Provider מבוקר: סופר prepare, אפשר לעכב propose, ולבחור heavy/stale_ok."""
    name = "fake"

    def __init__(self, heavy=False, stale_ok=False, result="box"):
        self.heavy, self.stale_ok = heavy, stale_ok
        self.prepares: list = []
        self.proposes = 0
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.result = result
        self.boom: BaseException | None = None

    def prepare(self, rgb, cancel=None):
        self.prepares.append(rgb.shape)
        return Context(self.name, rgb.shape[:2], data=len(self.prepares), nbytes=1)

    def propose(self, ctx, prompts, cancel=None):
        self.proposes += 1
        self.last_ctx = ctx
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.boom is not None:
            raise self.boom
        if self.result is None:
            return None
        return Proposal(np.full((3, 4), 255, np.uint8), (1, 1, 5, 4))


IMG = np.zeros((20, 30, 3), np.uint8)
PR = Prompts(points=(Point(1, 1),))


def svc(provider, **kw):
    return SelectionService(provider, **kw).start()


class TestService:
    def test_propose_prepares_once_and_caches(self):
        f = Fake()
        with svc(f) as s:
            r1 = s.propose(IMG, "k1", PR)
            assert r1.result(3).rect == (1, 1, 5, 4) and r1.state is RequestState.DONE
            s.propose(IMG, "k1", PR).result(3)
            assert len(f.prepares) == 1 and f.proposes == 2 and s.context_keys == ["k1"]

    def test_warm_then_propose_no_second_prepare(self):
        f = Fake()
        with svc(f) as s:
            w = s.warm(IMG, "k")
            assert w.result(3) is None and w.state is RequestState.DONE
            s.propose(IMG, "k", PR).result(3)
            assert len(f.prepares) == 1

    def test_lru_eviction(self):
        f = Fake()
        with svc(f, max_contexts=2) as s:
            for k in ("a", "b"):
                s.propose(IMG, k, PR).result(3)
            s.propose(IMG, "a", PR).result(3)                 # a חדש יותר מ-b
            s.propose(IMG, "c", PR).result(3)
            assert set(s.context_keys) == {"a", "c"}

    def test_invalidate(self):
        f = Fake()
        with svc(f) as s:
            s.propose(IMG, "a", PR).result(3)
            s.propose(IMG, "b", PR).result(3)
            s.invalidate("a")
            assert s.context_keys == ["b"]
            s.invalidate()
            assert s.context_keys == []

    def test_empty_result_is_not_error(self):
        f = Fake(result=None)
        with svc(f) as s:
            r = s.propose(IMG, "k", PR)
            assert r.result(3) is None and r.state is RequestState.EMPTY and r.error is None

    def test_provider_errors(self):
        f = Fake()
        with svc(f) as s:
            f.boom = SelectionError("קלט")
            r = s.propose(IMG, "k", PR)
            with pytest.raises(SelectionError, match="קלט"):
                r.result(3)
            assert r.state is RequestState.FAILED
            f.boom = RuntimeError("קרס")
            r2 = s.propose(IMG, "k", PR)
            with pytest.raises(SelectionError):
                r2.result(3)
            f.boom = None
            assert s.propose(IMG, "k", PR).result(3) is not None        # ה-Worker חי

    def test_new_propose_cancels_running_and_queued(self):
        f = Fake()
        f.gate = threading.Event()
        with svc(f) as s:
            a = s.propose(IMG, "k", PR)
            assert f.started.wait(3)
            b = s.propose(IMG, "k", PR)                          # ממתינה
            c = s.propose(IMG, "k", PR)                          # מבטלת את b ואת a
            assert b.state is RequestState.CANCELLED
            f.gate.set()
            assert a.wait(3) and c.wait(3)
            assert a.state is RequestState.CANCELLED and a.proposal is None     # הסתיים בפועל, אך לא נמסר
            with pytest.raises(CancelledError):
                a.result()
            assert c.state is RequestState.DONE and f.proposes == 2

    def test_explicit_cancel_and_callback(self):
        f = Fake()
        f.gate = threading.Event()
        seen = []
        with svc(f) as s:
            a = s.propose(IMG, "k", PR, on_done=seen.append)
            assert f.started.wait(3)
            assert a.cancel() is True
            f.gate.set()
            assert a.wait(3) and seen == [a] and a.state is RequestState.CANCELLED
            assert a.cancel() is False                            # כבר הסתיימה

    def test_callback_exception_does_not_kill_worker(self):
        f = Fake()
        with svc(f) as s:
            def bad(_r):
                raise RuntimeError("x")
            s.propose(IMG, "k", PR, on_done=bad).wait(3)
            assert s.propose(IMG, "k", PR).result(3) is not None

    def test_warm_replaces_older_warm_only(self):
        f = Fake(heavy=True)
        flag = {"busy": True}
        with svc(f, busy=lambda: flag["busy"], busy_poll=0.005) as s:
            w1 = s.warm(IMG, "k1")
            assert wait_until(lambda: w1.state is RequestState.RUNNING)
            w2 = s.warm(IMG, "k2")
            p = s.propose(IMG, "k2", PR)                        # propose לא מבטל warm
            assert w1.wait(3) and w1.state is RequestState.CANCELLED
            flag["busy"] = False
            assert w2.wait(3) and w2.state is RequestState.DONE and p.result(3) is not None

    def test_heavy_waits_for_busy_and_can_be_cancelled(self):
        f = Fake(heavy=True)
        flag = {"busy": True}
        with svc(f, busy=lambda: flag["busy"], busy_poll=0.005) as s:
            r = s.propose(IMG, "k", PR)
            time.sleep(0.1)
            assert r.state is RequestState.RUNNING and f.proposes == 0 and f.prepares == []
            flag["busy"] = False
            assert r.result(3) is not None
            flag["busy"] = True
            r2 = s.propose(IMG, "k", PR)
            time.sleep(0.05)
            r2.cancel()
            assert r2.wait(3) and r2.state is RequestState.CANCELLED

    def test_light_provider_ignores_busy(self):
        f = Fake(heavy=False)
        with svc(f, busy=lambda: True) as s:
            assert s.propose(IMG, "k", PR).result(3) is not None

    def test_stale_ok_uses_old_context_then_prepares_in_background(self):
        f = Fake(stale_ok=True)
        with svc(f) as s:
            s.propose(IMG, "v1", PR).result(3)
            r = s.propose(IMG, "v2", PR)
            assert r.result(3) is not None and r.stale is True and f.last_ctx.data == 1
            assert wait_until(lambda: "v2" in s.context_keys)      # Context מדויק הוכן ברקע
            r3 = s.propose(IMG, "v2", PR)
            r3.result(3)
            assert r3.stale is False and f.last_ctx.data == 2

    def test_stale_not_used_for_other_shape_or_when_disallowed(self):
        f = Fake(stale_ok=True)
        with svc(f) as s:
            s.propose(IMG, "v1", PR).result(3)
            r = s.propose(np.zeros((10, 10, 3), np.uint8), "v2", PR)
            r.result(3)
            assert r.stale is False and len(f.prepares) == 2
        g = Fake(stale_ok=False)
        with svc(g) as s:
            s.propose(IMG, "v1", PR).result(3)
            r = s.propose(IMG, "v2", PR)
            r.result(3)
            assert r.stale is False and len(g.prepares) == 2

    def test_new_propose_cancels_background_prepare(self):
        f = Fake(stale_ok=True)
        with svc(f) as s:
            s.propose(IMG, "v1", PR).result(3)
            f.gate = threading.Event()
            s.propose(IMG, "v2", PR)                              # נענית מ-stale ומשאירה warm ברקע
            f.gate.set()
            s.propose(IMG, "v3", PR).result(3)
            assert wait_until(lambda: "v3" in s.context_keys)

    def test_lifecycle(self):
        s = SelectionService(Fake())
        with pytest.raises(SchedulerClosedError):
            s.propose(IMG, "k", PR)
        s.start()
        s.start()
        f = s.provider
        f.gate = threading.Event()
        a = s.propose(IMG, "k", PR)
        assert f.started.wait(3)
        s.shutdown(wait=False)
        f.gate.set()
        assert a.wait(3) and a.state is RequestState.CANCELLED
        with pytest.raises(SchedulerClosedError):
            s.warm(IMG, "k")
        s.shutdown()

    def test_input_validation(self):
        with svc(Fake()) as s:
            with pytest.raises(SelectionError):
                s.propose(np.zeros((5, 5), np.uint8), "k", PR)
            with pytest.raises(SelectionError):
                s.propose(IMG, "k", "not-prompts")
        with pytest.raises(ValueError):
            SelectionService(Fake(), max_contexts=0)

    def test_request_result_timeout(self):
        f = Fake()
        f.gate = threading.Event()
        with svc(f) as s:
            r = s.propose(IMG, "k", PR)
            with pytest.raises(TimeoutError):
                r.result(0.05)
            f.gate.set()
            assert r.result(3) is not None


# ------------------------------------------------------------------------------------------ מקצה לקצה
def test_end_to_end_wand_service_to_mask_layer_and_undo():
    img = scene()
    layer = MaskLayer(*img.shape[:2])
    hist = MaskHistory(layer)
    with svc(MagicWandProvider()) as s:
        r = s.propose(img, ("s", 0), Prompts(points=(Point(80, 70),)))
        prop = r.result(3)
        assert prop.rect == (50, 40, 120, 100)
        hist.do(lambda m: prop.add_to(m))
        assert layer.bbox() == (50, 40, 120, 100) and layer.pixel_count == 70 * 60
        r2 = s.propose(img, ("s", 0), Prompts(points=(Point(200, 150),)))          # אותו Context
        hist.do(lambda m: r2.result(3).add_to(m))
        assert len(layer.components()) == 2 and s.context_keys == [("s", 0)]
        hist.undo()
        assert layer.pixel_count == 70 * 60
        hist.undo()
        assert layer.empty
