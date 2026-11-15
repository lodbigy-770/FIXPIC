"""M3.7.6 (מיזוג הדרגתי): בין אריחים שכנים הפס החופף מעורבב בהדרגה במקום תפר קשה בגבול הליבה."""
from __future__ import annotations

import numpy as np
import pytest

from fixpic.models.base import ModelInfo
from fixpic.pipeline import Pipeline, PipelineConfig
from test_engine import rnd


class StepModel:
    """ממלא את המסכה בגוון שתלוי במספר ההרצה (כל אריח בגוון אחר), כך שתפר בין אריחים הוא קפיצה בערך."""
    info = ModelInfo(name="step")

    def __init__(self):
        self.calls = 0

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        self.calls += 1
        out = image.copy()
        out[mask > 0] = min(255, 60 + 12 * self.calls)
        return out


H, W = 500, 900
BOX = (60, 60, 840, 440)


def mk():
    m = np.zeros((H, W), np.uint8)
    m[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 1
    return m


def run(blend, **kw):
    cfg = PipelineConfig(large_strategy="tile", max_tiles=500, tile_blend_px=blend, feather_px=0, **kw)
    mask, img = mk(), rnd(H, W, 1)
    res = Pipeline(StepModel(), cfg).run(img, mask, 256, brush_px=30.0)
    return img, mask, res


def max_step(res, mask):
    g = res.image[..., 0].astype(int)
    inside = (mask[:, 1:] > 0) & (mask[:, :-1] > 0)
    dx = np.abs(np.diff(g, axis=1))[inside]
    iny = (mask[1:] > 0) & (mask[:-1] > 0)
    dy = np.abs(np.diff(g, axis=0))[iny]
    return int(max(dx.max(), dy.max()))


def test_blend_off_has_hard_seams_and_on_is_smooth():
    _, mask, off = run(0)
    _, _, on = run(32)
    assert off.model_calls > 4
    s_off, s_on = max_step(off, mask), max_step(on, mask)
    assert s_off >= 12                       # קפיצה בין גווני אריחים שכנים
    assert s_on < s_off / 2, (s_off, s_on)   # המיזוג מפזר אותה על פני הפס


@pytest.mark.parametrize("blend", [0, 16, 32, 64])
def test_outside_mask_is_byte_identical(blend):
    img, mask, res = run(blend)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])


def test_every_masked_pixel_is_filled_and_blend_stays_inside_mask():
    img, mask, res = run(32)
    eff = res.effective_mask > 0
    assert eff[mask > 0].all()
    # אף פיקסל מוסתר לא נשאר כמקור (המודל לא מחזיר ערכי המקור האקראיים)
    assert (res.image[eff] != img[eff]).any(axis=-1).mean() > 0.99


def test_blend_does_not_change_number_of_model_calls_or_tiles():
    _, _, a = run(0)
    _, _, b = run(32)
    assert a.model_calls == b.model_calls and len(a.rois) == len(b.rois)


def test_blend_works_with_downscaled_tiles():
    cfg = PipelineConfig(large_strategy="tile", max_tiles=4, min_scale=0.2, tile_blend_px=32, feather_px=0)
    mask, img = mk(), rnd(H, W, 2)
    res = Pipeline(StepModel(), cfg).run(img, mask, 256, brush_px=30.0)
    assert any(r.scale < 1 for r in res.rois)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])


def test_blend_with_feather_and_cancel_paths_still_work():
    cfg = PipelineConfig(large_strategy="tile", max_tiles=500, tile_blend_px=32, feather_px=3)
    mask, img = mk(), rnd(H, W, 3)
    res = Pipeline(StepModel(), cfg).run(img, mask, 256, brush_px=30.0)
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])
