import tracemalloc

import numpy as np
import pytest

from fixpic.errors import CancelledError, InferenceError, OutOfMemoryError, is_oom
from fixpic.models import AutoModel, CancelToken
from fixpic.pipeline import Pipeline, PipelineConfig, estimate_memory_mb

from test_engine import FILL, FakeModel, rnd
from test_m24 import blocks


class OomAbove(FakeModel):
    """נכשל בזיכרון בכל קלט גדול מ-limit; סופר ניסיונות."""

    def __init__(self, limit, exc=OutOfMemoryError):
        super().__init__()
        self.limit, self.exc, self.attempts = limit, exc, []

    def run(self, image, mask, cancel=None):
        self.attempts.append(image.shape[0])
        if image.shape[0] > self.limit:
            raise self.exc("אין מספיק זיכרון (מדומה)")
        return super().run(image, mask, cancel)


# ------------------------------------------------------------------ OOM
@pytest.mark.parametrize("exc", [OutOfMemoryError, MemoryError])
def test_oom_retries_in_smaller_bucket_and_result_is_correct(exc):
    img = rnd(800, 800)
    mask = blocks(800, 800, (300, 360, 300, 380))
    fm = OomAbove(384, exc)
    res = Pipeline(fm, PipelineConfig(feather_px=0)).run(img, mask, 512)
    assert fm.attempts[:2] == [512, 384] and res.size_used == 384 and res.oom_retries == 1
    eff = res.effective_mask.astype(bool)
    assert (res.image[eff] == FILL).all() and np.array_equal(res.image[~eff], img[~eff])


def test_oom_at_every_size_raises_and_leaves_input_untouched():
    img = rnd(500, 500)
    before = img.copy()
    fm = OomAbove(0)
    with pytest.raises(OutOfMemoryError):
        Pipeline(fm).run(img, blocks(500, 500, (200, 240, 200, 240)), 512)
    assert fm.attempts == [512, 384, 256] and np.array_equal(img, before)


def test_oom_retry_disabled():
    fm = OomAbove(0)
    with pytest.raises(OutOfMemoryError):
        Pipeline(fm, PipelineConfig(oom_retry=False)).run(rnd(500, 500), blocks(500, 500, (200, 240, 200, 240)), 512)
    assert fm.attempts == [512]


def test_oom_mid_tiling_restores_group_and_recovers():
    """אריחים רבים: ה-OOM מגיע באריח מאוחר; הקבוצה מוחזרת למקור ומתחילה מחדש בגודל קטן, התוצאה נכונה."""
    class LateOom(FakeModel):
        def __init__(self):
            super().__init__(); self.n = 0

        def run(self, image, mask, cancel=None):
            self.n += 1
            if image.shape[0] == 512 and self.n == 3:
                raise OutOfMemoryError("מדומה")
            return super().run(image, mask, cancel)

    img = rnd(1500, 2000)
    mask = blocks(1500, 2000, (100, 1400, 100, 1900))
    cfg = PipelineConfig(large_strategy="tile", feather_px=0, mask_dilate_px=0, max_tiles=500)
    res = Pipeline(LateOom(), cfg).run(img, mask, 512)
    assert res.oom_retries == 1 and res.size_used == 384
    eff = res.effective_mask.astype(bool)
    assert (res.image[eff] == FILL).all() and np.array_equal(res.image[~eff], img[~eff])


def test_oom_does_not_switch_backend_in_automodel():
    class OomPrimary(FakeModel):
        def run(self, image, mask, cancel=None):
            raise OutOfMemoryError("מדומה")

    class Spy(FakeModel):
        called = False

        def run(self, *a, **k):
            Spy.called = True
            return super().run(*a, **k)

    am = AutoModel(primary=OomPrimary(), fallback=Spy())
    am._cfg, am._active, am.backend = None, am._primary, "openvino"
    with pytest.raises(OutOfMemoryError):
        am.run(rnd(256, 256), np.ones((256, 256), np.uint8))
    assert am.backend == "openvino" and not Spy.called


def test_is_oom_detection():
    assert is_oom(MemoryError()) and is_oom(RuntimeError("std::bad_alloc")) \
        and is_oom(RuntimeError("Failed to allocate 2GB")) and not is_oom(RuntimeError("shape mismatch"))
    assert not isinstance(OutOfMemoryError("x"), InferenceError)


# --------------------------------------------------------------- limit
def test_memory_limit_rejects_before_running_model():
    img = rnd(2000, 3000)
    fm = FakeModel()
    need = estimate_memory_mb(2000, 3000, 1600)
    with pytest.raises(OutOfMemoryError):
        Pipeline(fm, PipelineConfig(memory_limit_mb=need - 1)).run(img, blocks(2000, 3000, (10, 60, 10, 60)))
    assert fm.shapes == []
    Pipeline(fm, PipelineConfig(memory_limit_mb=need)).run(img, blocks(2000, 3000, (10, 60, 10, 60)))
    assert fm.shapes


@pytest.mark.parametrize("kw,rect", [({"large_strategy": "downscale"}, (100, 2900, 100, 3900)),
                                     ({"large_strategy": "tile", "max_tiles": 999}, (100, 2900, 100, 3900)),
                                     ({}, (1000, 1100, 1000, 1100))])
def test_estimate_is_an_upper_bound_of_measured_peak(kw, rect):
    h, w = 3000, 4000
    img = rnd(h, w)
    mask = blocks(h, w, rect)
    tracemalloc.start()
    tracemalloc.reset_peak()
    Pipeline(FakeModel(), PipelineConfig(**kw)).run(img, mask, 512)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    assert peak <= estimate_memory_mb(h, w, 0)


# ------------------------------------------------------------ progress
def test_progress_counts_match_model_calls_and_are_monotonic():
    ev = []
    img = rnd(1500, 2000)
    mask = blocks(1500, 2000, (100, 1400, 100, 1900))
    cfg = PipelineConfig(large_strategy="tile", feather_px=0, max_tiles=500)
    res = Pipeline(FakeModel(), cfg).run(img, mask, 512, progress=ev.append)
    inpaint = [e for e in ev if e.stage == "inpaint"]
    assert [e.done for e in inpaint] == list(range(1, res.model_calls + 1))
    assert all(e.total == res.model_calls for e in ev) and ev[-1].stage == "done" and ev[-1].done == ev[-1].total


def test_progress_multiple_groups_and_callback_errors_are_swallowed():
    img = rnd(2000, 2000)
    mask = blocks(2000, 2000, (100, 140, 100, 140), (1800, 1840, 1800, 1840))
    calls = []

    def bad(e):
        calls.append(e)
        raise RuntimeError("callback שבור")

    res = Pipeline(FakeModel()).run(img, mask, 256, progress=bad)
    assert res.groups == 2 and len([c for c in calls if c.stage == "inpaint"]) == 2


def test_cancel_from_progress_callback_stops_between_tiles():
    tok = CancelToken()
    fm = FakeModel()
    img = rnd(1500, 2000)
    before = img.copy()
    mask = blocks(1500, 2000, (100, 1400, 100, 1900))
    cfg = PipelineConfig(large_strategy="tile", feather_px=0, max_tiles=500)

    def cb(e):
        if e.done == 2:
            tok.cancel()

    with pytest.raises(CancelledError):
        Pipeline(fm, cfg).run(img, mask, 512, cancel=tok, progress=cb)
    assert len(fm.shapes) == 2 and np.array_equal(img, before)


def test_progress_total_grows_after_oom_retry():
    ev = []
    fm = OomAbove(384)
    Pipeline(fm, PipelineConfig(feather_px=0)).run(rnd(800, 800), blocks(800, 800, (300, 360, 300, 380)), 512,
                                                   progress=ev.append)
    assert ev[-1].done <= ev[-1].total and ev[-1].stage == "done"
