"""M2.8: הרחבת מסכה אוטומטית לפי גודל המברשת (ADR-008) ורזולוציה מקורית תחילה (ARCHITECTURE §4ב)."""
import cv2
import numpy as np
import pytest

from fixpic.engine import JobScheduler
from fixpic.errors import InvalidInputError
from fixpic.models import ModelInfo
from fixpic.pipeline import Pipeline, PipelineConfig, plan_roi
from fixpic.pipeline.dilate import dilate_radius, estimate_brush_px


def rnd(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def stroke(h, w, width, length=300, y=None, x0=100):
    """משיכה אופקית בעובי width (פיקסלים)."""
    y = h // 2 if y is None else y
    m = np.zeros((h, w), np.uint8)
    m[y:y + width, x0:x0 + length] = 1
    return m


class Spy:
    info = ModelInfo(name="spy")

    def __init__(self):
        self.inputs = []

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        self.inputs.append((image.copy(), mask.copy()))
        out = image.copy()
        out[mask > 0] = (1, 2, 3)
        return out


# -------------------------------------------------------- הערכת מברשת
@pytest.mark.parametrize("width", [1, 4, 10, 40, 100])
def test_estimate_brush_matches_stroke_width(width):
    est = estimate_brush_px(stroke(1000, 1000, width, length=600))
    assert width * 0.85 <= est <= width * 1.25 + 1.0    # קו של פיקסל אחד נותן 2: כל פיקסליו פיקסלי שפה


def test_estimate_empty_and_single_pixel_and_two_strokes():
    assert estimate_brush_px(np.zeros((50, 50), np.uint8)) == 0.0
    one = np.zeros((50, 50), np.uint8)
    one[10, 10] = 1
    assert 1 <= estimate_brush_px(one) <= 3
    two = stroke(1000, 1000, 20, y=100) | stroke(1000, 1000, 20, y=800)
    assert 17 <= estimate_brush_px(two) <= 25


def test_dilate_radius_rules():
    assert dilate_radius(0, 99, 0.125, 3, 20) == 0                 # כבוי
    assert dilate_radius(7, 99, 0.125, 3, 20) == 7                 # קבוע: מתעלם מהמברשת
    assert dilate_radius(None, 40, 0.125, 3, 20) == 5              # יחסי
    assert dilate_radius(None, 4, 0.125, 3, 20) == 3               # מינימום
    assert dilate_radius(None, 1000, 0.125, 3, 20) == 20           # מקסימום
    assert dilate_radius(None, 0, 0.125, 0, 20) == 0               # מינימום 0 מאפשר כיבוי אוטומטי


# ------------------------------------------------------ הרחבה ב-Pipeline
def ellipse_dilate(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k)


@pytest.mark.parametrize("width,expected", [(10, 3), (40, 5), (80, 9), (400, 20)])   # 80: משיכה קצרה מוערכת ב-69 (הטיית קצוות)
def test_auto_dilate_scales_with_estimated_brush(width, expected):
    img = rnd(1000, 1000)
    m = stroke(1000, 1000, width, length=500, y=300, x0=200)
    res = Pipeline(Spy(), PipelineConfig(large_strategy="downscale")).run(img, m * 255, 512)
    assert res.dilate_px == expected
    assert np.array_equal(res.effective_mask, ellipse_dilate(m, expected))


def test_ui_brush_overrides_estimate_and_fixed_overrides_both():
    img = rnd(600, 600)
    m = stroke(600, 600, 10, length=200, y=300, x0=200)
    pl = Pipeline(Spy())
    assert pl.run(img, m, 256, brush_px=80).dilate_px == 10        # המברשת מה-UI גוברת על הערכה (3)
    assert pl.run(img, m, 256).dilate_px == 3
    fixed = Pipeline(Spy(), PipelineConfig(mask_dilate_px=0)).run(img, m, 256, brush_px=80)
    assert fixed.dilate_px == 0 and np.array_equal(fixed.effective_mask, m)
    seven = Pipeline(Spy(), PipelineConfig(mask_dilate_px=7)).run(img, m, 256)
    assert seven.dilate_px == 7 and seven.brush_px == 0.0          # קבוע: לא מעריכים


def test_outside_effective_mask_identical_with_auto_dilate():
    img = rnd(500, 700, 3)
    m = stroke(500, 700, 30, length=300, y=200, x0=150)
    res = Pipeline(Spy()).run(img, m, 384, brush_px=30)
    eff = res.effective_mask.astype(bool)
    assert eff.sum() > m.sum()
    assert np.array_equal(res.image[~eff], img[~eff])


def test_invalid_config_rejected():
    for bad in (dict(dilate_max=1, dilate_min=3), dict(mask_dilate_px=-1), dict(dilate_ratio=-0.1),
                dict(native_margin_ratio=0.5)):
        with pytest.raises(InvalidInputError):
            Pipeline(Spy(), PipelineConfig(**bad))


def test_scheduler_passes_brush_px():
    img = rnd(400, 400)
    m = stroke(400, 400, 10, length=100, y=200, x0=150)
    with JobScheduler(Pipeline(Spy())) as s:
        res = s.submit(img, m, 256, brush_px=120).result(10)
    assert res.brush_px == 120 and res.dilate_px == 15


# ------------------------------------------------ רזולוציה מקורית תחילה
def box(h, w, side, cy=1000, cx=1000):
    m = np.zeros((h, w), np.uint8)
    m[cy:cy + side, cx:cx + side] = 1
    return m


def test_native_first_shrinks_context_before_downscaling():
    m = box(2000, 2000, 300)                       # הקשר מלא: 300+2·75=450 > 384; מצומצם: 300+2·30=360 ≤ 384
    old = plan_roi(m, 384)
    new = plan_roi(m, 384, native_margin_ratio=0.10)
    assert old.downscaled and old.scale < 1
    assert not new.downscaled and new.scale == 1.0 and new.width == new.height == 384


def test_native_first_still_downscales_when_even_tight_context_does_not_fit():
    m = box(2000, 2000, 360)                       # 360+2·36=432 > 384
    assert plan_roi(m, 384, native_margin_ratio=0.10).downscaled


def test_native_first_respects_min_margin():
    m = box(2000, 2000, 340)
    assert not plan_roi(m, 384, min_margin=16, native_margin_ratio=0.0).downscaled   # 340+2·16=372 ≤ 384
    assert plan_roi(m, 384, min_margin=30, native_margin_ratio=0.0).downscaled       # 340+2·30=400 > 384


def test_pipeline_native_first_sends_unscaled_pixels_and_off_switch_restores_downscale():
    img = rnd(2000, 2000, 4)
    m = box(2000, 2000, 300) * 255
    spy = Spy()
    res = Pipeline(spy, PipelineConfig(mask_dilate_px=0, feather_px=0)).run(img, m, 384)
    p = res.roi
    assert not p.downscaled and (p.width, p.height) == (384, 384)
    sent, _ = spy.inputs[0]
    assert np.array_equal(sent, img[p.y0:p.y1, p.x0:p.x1])         # הפיקסלים נשלחו כמו שהם, בלי Resize
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])

    off = Pipeline(spy, PipelineConfig(mask_dilate_px=0, feather_px=0, native_margin_ratio=0.25)).run(img, m, 384)
    assert off.roi.downscaled                                      # כיבוי = התנהגות קודמת


def test_cli_brush_px_and_json_report(tmp_path, monkeypatch, capsys):
    from fixpic import cli
    d = tmp_path / "תיקייה"
    d.mkdir()
    cv2.imencode(".png", rnd(300, 400)[..., ::-1])[1].tofile(str(d / "a.png"))
    cv2.imencode(".png", stroke(300, 400, 10, length=100, y=150, x0=150) * 255)[1].tofile(str(d / "m.png"))
    monkeypatch.setattr(cli, "_load_model", lambda a: Spy())
    assert cli.main([str(d / "a.png"), str(d / "m.png"), "--brush-px", "80", "--json"]) == 0
    info = __import__("json").loads(capsys.readouterr().out)
    assert info["dilate_px"] == 10 and info["brush_px"] == 80.0
    assert cli.main([str(d / "a.png"), str(d / "m.png"), "--dilate", "0", "--json"]) == 0
    assert __import__("json").loads(capsys.readouterr().out)["dilate_px"] == 0
