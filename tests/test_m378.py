"""M3.7.8: שחזור טקסטורה (`pipeline/texture.py`) מחזיר גרעין לאזור מולא מוקטן, מותאם לסביבה, רק בתוך המסכה."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from fixpic.errors import CancelledError, InvalidInputError
from fixpic.models.base import CancelToken, ModelInfo
from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.pipeline.texture import restore_texture
from fixpic.quality import texture_ratio

H, W = 900, 1400
BOX = (100, 100, 1300, 800)


def background(seed=1, amp=2.0):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    base = 100 + 40 * np.sin(xx / 170) + 30 * np.cos(yy / 230)
    n = cv2.GaussianBlur(r.normal(0, 7, (H, W)).astype(np.float32), (0, 0), 1.0) * amp
    return np.clip(np.stack([base + n + i * 10 for i in range(3)], -1), 0, 255).astype(np.uint8)


def mk(box=BOX):
    m = np.zeros((H, W), np.uint8)
    m[box[1]:box[3], box[0]:box[2]] = 1
    return m


def hp_rms(img, sel):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.sqrt(((g - cv2.GaussianBlur(g, (0, 0), 1.5))[sel] ** 2).mean()))


class Smooth:
    """"מריחה": ממלא את המסכה בגרסה מטושטשת מאוד של הקלט (בלי טקסטורה)."""
    info = ModelInfo(name="smooth")

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        out = image.copy()
        b = cv2.GaussianBlur(image, (0, 0), 20)
        out[mask > 0] = b[mask > 0]
        return out


class Shift(Smooth):
    """מילוי עם טקסטורה אמיתית (הקלט מוזז), כלומר בלי חוסר."""

    def run(self, image, mask, cancel=None):
        out = image.copy()
        sh = np.roll(image, 200, axis=1)
        out[mask > 0] = sh[mask > 0]
        return out


def cfg(**kw):
    kw.setdefault("large_strategy", "downscale")
    kw.setdefault("feather_px", 3)
    return PipelineConfig(**kw)


def run(model=None, img=None, mask=None, size=256, **kw):
    img = background() if img is None else img
    mask = mk() if mask is None else mask
    return img, Pipeline(model or Smooth(), cfg(**kw)).run(img, mask, size, brush_px=30.0)


@pytest.fixture(scope="module")
def pair():
    img, off = run()
    _, on = run(texture_restore=True)
    return img, off, on


def test_off_by_default():
    c = PipelineConfig()
    assert c.texture_restore is False
    _, a = run()
    _, b = run(texture_restore=False)
    assert np.array_equal(a.image, b.image) and a.texture_groups == 0


def test_restores_texture_ratio_without_overshoot(pair):
    img, off, on = pair
    before = texture_ratio(off.image, off.effective_mask)[0]
    after = texture_ratio(on.image, on.effective_mask)[0]
    assert before < 0.4 and 0.8 < after < 1.25
    assert on.texture_groups == 1


def test_only_masked_pixels_change_and_outside_is_byte_identical(pair):
    img, off, on = pair
    eff = on.effective_mask > 0
    assert np.array_equal(on.image[~eff], img[~eff])
    assert (on.image != off.image).any()
    assert not (on.image != off.image).any(axis=2)[~eff].any()


def test_grain_is_luminance_only(pair):
    _, off, on = pair
    d = on.image.astype(np.int16) - off.image.astype(np.int16)
    mid = (off.image.min(axis=2) > 20) & (off.image.max(axis=2) < 235)       # בלי חיתוך בקצוות
    assert np.array_equal(d[..., 0][mid], d[..., 1][mid]) and np.array_equal(d[..., 1][mid], d[..., 2][mid])


def test_deterministic_seed_and_strength():
    _, a = run(texture_restore=True, texture_seed=3)
    _, b = run(texture_restore=True, texture_seed=3)
    _, c = run(texture_restore=True, texture_seed=4)
    _, off = run()
    assert np.array_equal(a.image, b.image) and not np.array_equal(a.image, c.image)
    _, half = run(texture_restore=True, texture_strength=0.5)
    _, zero = run(texture_restore=True, texture_strength=0.0)
    eff = a.effective_mask > 0
    assert np.array_equal(zero.image, off.image) and zero.texture_groups == 0
    d1 = np.abs(a.image.astype(float) - off.image)[eff].mean()
    d5 = np.abs(half.image.astype(float) - off.image)[eff].mean()
    assert 0.3 * d1 < d5 < 0.7 * d1


def test_adapts_to_surroundings_texture():
    r = np.random.default_rng(5)
    n = cv2.GaussianBlur(r.normal(0, 7, (H, W)).astype(np.float32), (0, 0), 1.0)
    amp = np.where(np.arange(W)[None, :] < 700, 4.0, 0.4).astype(np.float32)       # שמאל עשיר, ימין כמעט חלק
    img = np.clip(np.repeat((120 + n * amp)[..., None], 3, 2), 0, 255).astype(np.uint8)
    _, res = run(img=img, mask=mk((400, 100, 1000, 800)), texture_restore=True)
    eff = res.effective_mask > 0
    cols = np.arange(W)[None, :].repeat(H, 0)
    left, right = hp_rms(res.image, eff & (cols < 560)), hp_rms(res.image, eff & (cols > 840))
    assert res.texture_groups == 1 and left > 2.0 * right


def test_flat_surroundings_get_no_grain():
    flat = np.full((H, W, 3), 120, np.uint8)
    _, off = run(img=flat)
    _, on = run(img=flat, texture_restore=True)
    assert on.texture_groups == 0 and np.array_equal(on.image, off.image)


def test_guard_skips_when_fill_already_textured():
    # מילוי ב-native (בלי הקטנה) שמכיל טקסטורה אמיתית; texture_max_scale גבוה כדי שהשלב ירוץ ויחליט לפי המגן
    box = (300, 250, 700, 550)
    img, off = run(Shift(), mask=mk(box), size=512)
    _, on = run(Shift(), mask=mk(box), size=512, texture_restore=True, texture_max_scale=2.0)
    assert texture_ratio(off.image, off.effective_mask)[0] > 0.85
    assert on.texture_groups == 0 and np.array_equal(on.image, off.image)


def test_scale_gating():
    box = (300, 250, 360, 310)                                                      # קטן: native, קנה מידה 1.0
    img, a = run(mask=mk(box), size=256)
    _, b = run(mask=mk(box), size=256, texture_restore=True)
    assert b.texture_groups == 0 and np.array_equal(a.image, b.image)
    _, c = run(texture_restore=True, texture_max_scale=0.1)                         # גדול (קנה מידה כ-0.2) מעל הסף
    assert c.texture_groups == 0
    _, d = run(texture_restore=True)
    assert d.texture_groups == 1


def test_fill_scale_follows_coarse_pass_and_tiles():
    pipe = Pipeline(Smooth(), cfg(coarse_fine=True, coarse_size=None, large_strategy="tile", max_tiles=500))
    from fixpic.pipeline.grouping import Group
    g = Group(BOX, mk()[BOX[1]:BOX[3], BOX[0]:BOX[2]])
    s_def = pipe._fill_scale(g, 256, 256, (H, W))
    pipe2 = Pipeline(Smooth(), cfg(coarse_fine=True, large_strategy="tile", max_tiles=500, coarse_size=512))
    s_512 = pipe2._fill_scale(g, 256, 256, (H, W))
    assert s_def < 0.3 < s_512 and s_512 == pytest.approx(2 * s_def, rel=0.1)
    tiles = Pipeline(Smooth(), cfg(coarse_fine=False, large_strategy="tile", max_tiles=500))
    assert tiles._fill_scale(g, 256, 256, (H, W)) == 1.0                              # אריחים במקור: אין חוסר
    assert tiles._fill_scale(g, 256, 256, (H, W)) == 1.0


def test_coarse_fine_path_restores_interior():
    _, off = run(coarse_fine=True, coarse_belt_px=64, coarse_size=None, large_strategy="tile", max_tiles=500, tile_blend_px=0)
    _, on = run(coarse_fine=True, coarse_belt_px=64, coarse_size=None, large_strategy="tile", max_tiles=500, tile_blend_px=0, texture_restore=True)
    assert on.texture_groups == 1
    assert texture_ratio(on.image, on.effective_mask)[0] > texture_ratio(off.image, off.effective_mask)[0] + 0.3


def test_mask_at_image_corner_and_edges():
    img = background()
    res = Pipeline(Smooth(), cfg(texture_restore=True)).run(img, mk((0, 0, 900, 600)), 256, brush_px=30.0)
    eff = res.effective_mask > 0
    assert res.texture_groups == 1 and np.array_equal(res.image[~eff], img[~eff])


def test_direct_call_edge_cases_and_cancel():
    img = background()
    m = mk()
    work = img.copy()
    assert not restore_texture(work, m, BOX, np.zeros((700, 1200), np.uint8)).applied              # קבוצה ריקה
    assert not restore_texture(work, m, BOX, m[100:800, 100:1300], strength=0).applied
    full = np.ones((H, W), np.uint8)                                                                # אין סביבה
    assert restore_texture(img.copy(), full, (0, 0, W, H), full).reason == "אין סביבה למדידה"
    tok = CancelToken()
    tok.cancel()
    smooth = Smooth().run(img, m)
    before = smooth.copy()
    with pytest.raises(CancelledError):
        restore_texture(smooth, m, BOX, m[100:800, 100:1300], cancel=tok)
    assert np.array_equal(smooth, before)                                                           # בוטל לפני כל שינוי


def test_info_reports_ratio_and_amplitude():
    img = background()
    m = mk()
    work = Smooth().run(img, m)
    info = restore_texture(work, m, BOX, m[100:800, 100:1300])
    assert info.applied and info.ratio < 0.4 and 1.0 < info.amp_rms < 6.0


def test_invalid_config():
    for kw in ({"texture_strength": -1}, {"texture_max_scale": 0}, {"texture_guard": 0}):
        with pytest.raises(InvalidInputError):
            Pipeline(Smooth(), PipelineConfig(**kw))


def test_bench_grid_has_texture_axis():
    spec = importlib.util.spec_from_file_location("qb378", Path(__file__).resolve().parents[1] / "tools" / "m3" / "quality_bench.py")
    qb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qb)
    g = qb.build_grid([384], ["downscale"], [0.5], [3], [64], [32], textures=[0, 1])
    assert [c["texture_restore"] for c in g] == [0, 1] and g[1]["id"].endswith("-tx") and not g[0]["id"].endswith("-tx")
    assert len(qb.build_grid([384], ["downscale"], [0.5], [3], [64], [32])) == 1      # ברירת מחדל: ללא הציר


def test_default_adr012_path_restores_texture_at_coarse_scale():
    """ברירות המחדל של M3.7.9 (מגס-לדק, חגורה 0, coarse_size=768): חלון גס יחיד, והשחזור מכוון לקנה המידה שלו."""
    img = background()
    off = Pipeline(Smooth(), PipelineConfig(feather_px=3)).run(img, mk(), 256, brush_px=30.0)
    on = Pipeline(Smooth(), PipelineConfig(feather_px=3, texture_restore=True)).run(img, mk(), 256, brush_px=30.0)
    assert off.model_calls == on.model_calls == 1 and on.texture_groups == 1
    assert texture_ratio(on.image, on.effective_mask)[0] > texture_ratio(off.image, off.effective_mask)[0] + 0.3
    eff = on.effective_mask > 0
    assert np.array_equal(on.image[~eff], img[~eff])
