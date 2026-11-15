"""M3.7.7 (מגס לדק): מעבר גס בחלון יחיד מוקטן, ואז אריחים רק על חגורה לאורך גבול המסכה."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fixpic.errors import CancelledError
from fixpic.models.base import CancelToken, ModelInfo
from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.pipeline.grouping import Group
from test_engine import FakeModel, rnd

H, W = 900, 1400
BOX = (100, 100, 1300, 800)


class CallModel:
    """ממלא את המסכה בערך שתלוי במספר ההרצה: הרצה 1 (המעבר הגס) = 50, הבאות 100, 110, ..."""
    info = ModelInfo(name="call")

    def __init__(self):
        self.calls = 0
        self.masks = []

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        self.calls += 1
        self.masks.append(int((mask > 0).sum()))
        out = image.copy()
        out[mask > 0] = 50 if self.calls == 1 else min(255, 90 + 10 * self.calls)
        return out


def mk(box=BOX):
    m = np.zeros((H, W), np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 1
    return m


def cfg(cf, **kw):
    kw.setdefault("feather_px", 0)
    return PipelineConfig(large_strategy="tile", max_tiles=500, coarse_fine=cf, **kw)


def run(cf, model=None, mask=None, **kw):
    mask = mk() if mask is None else mask
    img = rnd(H, W, 1)
    model = model or CallModel()
    return img, mask, model, Pipeline(model, cfg(cf, **kw)).run(img, mask, 256, brush_px=30.0)


def test_off_by_default_and_unchanged():
    assert PipelineConfig().coarse_fine is False
    _, _, _, a = run(False)
    _, _, _, b = run(False)
    assert np.array_equal(a.image, b.image) and a.model_calls > 4


def test_fewer_runs_first_is_coarse_then_belt_tiles():
    _, _, _, off = run(False)
    _, _, _, on = run(True)
    assert on.model_calls < off.model_calls
    assert on.rois[0].scale < 1 and on.rois[0].downscaled          # המעבר הגס: חלון יחיד מוקטן
    assert all(r.scale >= on.rois[1].scale for r in on.rois[1:])      # שאר ההרצות: אריחים


def test_interior_is_coarse_and_belt_is_refined_and_outside_untouched():
    img, mask, _, res = run(True, tile_blend_px=0)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])
    g = res.image[..., 0].astype(int)
    cy, cx = (BOX[1] + BOX[3]) // 2, (BOX[0] + BOX[2]) // 2
    assert g[cy, cx] == 50                                    # עמוק בפנים: המילוי הגס בלבד
    edge = [g[cy, BOX[0] + 3], g[BOX[1] + 3, cx], g[cy, BOX[2] - 4]]
    assert all(v > 50 for v in edge), edge                    # סמוך לגבול: מעבר עדין


def test_plan_matches_run_and_progress_total():
    mask, img = mk(), rnd(H, W, 2)
    pipe = Pipeline(CallModel(), cfg(True))
    rep = pipe.plan(mask, 256, 30.0)
    seen = []
    res = pipe.run(img, mask, 256, brush_px=30.0, progress=seen.append)
    assert rep.runs == res.model_calls == len(res.rois)
    assert seen[-1].stage == "done" and seen[-1].done == seen[-1].total == res.model_calls


def test_belt_geometry_and_image_edge():
    pipe = Pipeline(CallModel(), cfg(True, coarse_belt_px=20))
    m = np.zeros((300, 500), np.uint8)
    m[:, :400] = 1                                            # נוגע בקצה השמאלי, העליון והתחתון של התמונה
    g = Group((0, 0, 400, 300), m[:, :400].copy())
    belt, gain = pipe._belt(g, 1.0, m.shape)
    assert belt[150, 0] == 0 and belt[0, 100] == 0           # בקצה התמונה אין גבול
    assert belt[150, 399] == 1 and belt[150, 381] == 1        # בצד הימני יש (מחוץ לו אין מסכה)
    assert belt[150, 360] == 0 and belt[150, 200] == 0
    assert gain[150, 399] == 1.0 and gain[150, 390] == 1.0 and gain[150, 379] < 0.15
    assert gain.dtype == np.float32 and ((0 <= gain) & (gain <= 1)).all()


def test_tile_seams_only_inside_belt():
    mask = mk()
    off = Pipeline(FakeModel(), cfg(False)).tile_seams(mask, 256, 30.0)
    on = Pipeline(FakeModel(), cfg(True)).tile_seams(mask, 256, 30.0)
    assert len(off) > 0
    cov_len = lambda ls: sum(b - a for _, _, a, b in ls)     # noqa: E731
    assert cov_len(on) < cov_len(off)


def test_native_and_downscale_groups_unaffected():
    small = np.zeros((H, W), np.uint8)
    small[100:140, 100:160] = 1
    img = rnd(H, W, 3)
    a = Pipeline(CallModel(), cfg(False)).run(img, small, 256, brush_px=20.0)
    b = Pipeline(CallModel(), cfg(True)).run(img, small, 256, brush_px=20.0)
    assert np.array_equal(a.image, b.image) and a.model_calls == b.model_calls == 1
    c1 = PipelineConfig(large_strategy="downscale", coarse_fine=False)
    c2 = PipelineConfig(large_strategy="downscale", coarse_fine=True)
    r1 = Pipeline(CallModel(), c1).run(img, mk(), 256, brush_px=30.0)
    r2 = Pipeline(CallModel(), c2).run(img, mk(), 256, brush_px=30.0)
    assert np.array_equal(r1.image, r2.image) and r2.model_calls == 1


def test_cancel_between_passes_raises_and_input_unchanged():
    img, mask = rnd(H, W, 4), mk()
    orig = img.copy()
    tok = CancelToken()

    class Cancelling(CallModel):
        def run(self, image, mask, cancel=None):
            out = super().run(image, mask, cancel)
            if self.calls == 2:
                tok.cancel()
            return out

    with pytest.raises(CancelledError):
        Pipeline(Cancelling(), cfg(True)).run(img, mask, 256, brush_px=30.0, cancel=tok)
    assert np.array_equal(img, orig)


@pytest.mark.parametrize("feather", [0, 3, 8])
def test_feather_and_blend_keep_outside_identical(feather):
    img, mask, _, res = run(True, feather_px=feather, tile_blend_px=32)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])
    assert (res.image[eff] != img[eff]).any(axis=-1).mean() > 0.99   # כל פיקסל מוסתר שונה מהמקור
