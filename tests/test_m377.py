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


# ------------------------------------------------------------------ coarse_size (bucket נפרד למעבר הגס) + hook אבחון
class ShapeModel(CallModel):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def run(self, image, mask, cancel=None):
        self.shapes.append(image.shape[0])
        return super().run(image, mask, cancel)


def run_cs(cs, model=None, size=256, **kw):
    mask, img = mk(), rnd(H, W, 5)
    c = cfg(True, sizes=(256, 384, 512), coarse_size=cs, **kw)
    model = model or ShapeModel()
    return img, mask, model, Pipeline(model, c).run(img, mask, size, brush_px=30.0)


def test_coarse_size_applies_only_to_coarse_pass():
    _, _, model, res = run_cs(512)
    assert model.shapes[0] == 512 and set(model.shapes[1:]) == {256}
    assert res.rois[0].size == 512 and all(r.size == 256 for r in res.rois[1:])
    _, _, base, _ = run_cs(None)
    assert set(base.shapes) == {256}


def test_coarse_size_ignored_without_coarse_fine_and_validated_with_it():
    mask, img = mk(), rnd(H, W, 6)
    m = ShapeModel()
    Pipeline(m, PipelineConfig(large_strategy="tile", max_tiles=500, coarse_fine=False, coarse_size=512)).run(img, mask, 256, brush_px=30.0)
    assert set(m.shapes) == {256}
    from fixpic.errors import InvalidInputError
    with pytest.raises(InvalidInputError):
        Pipeline(ShapeModel(), cfg(True, coarse_size=1024)).run(img, mask, 256, brush_px=30.0)


def test_coarse_size_oom_falls_back_to_tile_bucket():
    class OomBig(ShapeModel):
        def run(self, image, mask, cancel=None):
            if image.shape[0] >= 512:
                from fixpic.errors import OutOfMemoryError
                raise OutOfMemoryError("מדומה")
            return super().run(image, mask, cancel)

    img, mask, model, res = run_cs(512, model=OomBig(), size=384)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff]) and res.oom_retries == 1
    assert max(model.shapes) <= 384 and res.size_used == 256


def test_coarse_hook_gets_coarse_only_crop():
    mask, img = mk(), rnd(H, W, 7)
    pipe = Pipeline(CallModel(), cfg(True, tile_blend_px=0))
    seen = []
    pipe.coarse_hook = lambda box, crop: seen.append((box, crop))
    res = pipe.run(img, mask, 256, brush_px=30.0)
    (box, crop), = seen
    x0, y0, x1, y1 = box
    assert crop.shape == (y1 - y0, x1 - x0, 3)
    cy, cx = (y1 - y0) // 2, (x1 - x0) // 2
    assert (crop[cy, cx] == 50).all()                         # אחרי המעבר הגס בלבד: ערך ההרצה הראשונה
    belt_px = res.image[y0 + 3, (x0 + x1) // 2]               # בתוצאה הסופית הקצה עודכן במעבר העדין
    assert (belt_px != 50).any()


def test_bench_grid_has_coarse_size_axis():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("qb377", Path(__file__).resolve().parents[1] / "tools" / "m3" / "quality_bench.py")
    qb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qb)
    g = qb.build_grid([384], ["auto"], [0.5], [3], [64], [32], coarses=[0, 1], coarse_sizes=[0, 768, 1024])
    ids = [c["id"] for c in g]
    assert len(ids) == 4 and any(i.endswith("-cf768") for i in ids) and any(i.endswith("-cf1024") for i in ids)
    assert sum(1 for c in g if not c["coarse_fine"]) == 1 and all(c["coarse_size"] == 0 for c in g if not c["coarse_fine"])
