import cv2
import numpy as np
import pytest

from fixpic.errors import CancelledError, EmptyMaskError, InvalidInputError, ModelLoadError
from fixpic.models import CancelToken, ModelConfig, ModelInfo
from fixpic.models.ort_model import OrtModel
from fixpic.pipeline import Pipeline, PipelineConfig, plan_roi
from fixpic.pipeline.prepost import postprocess, preprocess

FILL = np.array([10, 200, 30], np.uint8)


class FakeModel:
    """ממלא את המסכה בצבע קבוע ומשבש בכוונה גם מחוץ לה, כדי להוכיח שהמיזוג זורק את השיבוש."""
    info = ModelInfo(name="fake")

    def __init__(self):
        self.shapes = []

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        self.shapes.append((image.shape, mask.shape))
        out = (image.astype(int) + 7).clip(0, 255).astype(np.uint8)
        out[mask > 0] = FILL
        return out


def rnd(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


@pytest.mark.parametrize("hw", [(300, 400), (100, 90), (1000, 1500), (257, 259)])
@pytest.mark.parametrize("size", [256, 384, 512])
def test_outside_identical_inside_filled_fixed_shape(hw, size):
    h, w = hw
    img = rnd(h, w)
    mask = box(h, w, h // 3, h // 3 + max(4, h // 8), w // 4, w // 4 + max(4, w // 8))
    fm = FakeModel()
    before = img.copy()
    res = Pipeline(fm).run(img, mask, size)
    assert np.array_equal(img, before)                       # הקלט לא משתנה
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])        # מחוץ למסכה זהה לבייט
    if res.roi.downscaled:   # על הגבול ההגדלה חזרה מערבבת עם הסביבה; בודקים את פנים המסכה
        inner = cv2.erode(res.effective_mask, np.ones((int(6 / res.roi.scale) + 3,) * 2, np.uint8)).astype(bool)
        assert inner.any() and np.abs(res.image[inner].astype(int) - FILL).max() <= 1
    else:   # Feathering משאיר את רצועת הקצה (3px) מעורבבת; הפנים הוא פלט המודל
        inner = cv2.erode(res.effective_mask, np.ones((9, 9), np.uint8)).astype(bool)
        assert inner.any() and (res.image[inner] == FILL).all()
    assert all(s == ((size, size, 3), (size, size)) for s in fm.shapes)   # רק צורות קבועות


def test_large_mask_downscales_and_still_exact_outside():
    img = rnd(2000, 3000)
    mask = box(2000, 3000, 200, 1700, 300, 2500)
    res = Pipeline(FakeModel(), PipelineConfig(large_strategy="downscale")).run(img, mask, 512)
    assert res.roi.downscaled and res.roi.scale < 1
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])
    inner = cv2.erode(res.effective_mask, np.ones((int(6 / res.roi.scale) + 3,) * 2, np.uint8)).astype(bool)
    assert np.abs(res.image[inner].astype(int) - FILL).max() <= 1   # פנים המסכה: צבע אחיד אחרי resize


def test_native_resolution_when_context_fits():
    p = plan_roi(box(1000, 1000, 400, 450, 400, 450), 384)
    assert p.scale == 1.0 and p.width == p.height == 384


def test_roi_clamped_inside_image_at_corner():
    p = plan_roi(box(500, 500, 0, 20, 0, 20), 256)
    assert p.x0 == 0 and p.y0 == 0 and p.x1 <= 500 and p.y1 <= 500


def test_empty_mask_and_bad_inputs_and_bad_size():
    pl = Pipeline(FakeModel())
    img = rnd(100, 100)
    with pytest.raises(EmptyMaskError):
        pl.run(img, np.zeros((100, 100), np.uint8))
    with pytest.raises(InvalidInputError):
        pl.run(img, np.ones((50, 50), np.uint8))
    with pytest.raises(InvalidInputError):
        pl.run(img.astype(np.float32), np.ones((100, 100), np.uint8))
    with pytest.raises(InvalidInputError):
        pl.run(img, np.ones((100, 100), np.uint8), size=300)


def test_cancel_leaves_input_untouched():
    img = rnd(200, 200)
    before = img.copy()
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(CancelledError):
        Pipeline(FakeModel()).run(img, box(200, 200, 50, 90, 50, 90), cancel=tok)
    assert np.array_equal(img, before)


def test_dilate_off_keeps_mask_exact():
    img = rnd(300, 300)
    mask = box(300, 300, 100, 120, 100, 120)
    res = Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=0)).run(img, mask)
    assert np.array_equal(res.effective_mask, (mask > 0).astype(np.uint8))


def test_prepost_pad_crop_truncation():
    img = rnd(37, 53)
    rgb, m, hw = preprocess(img, np.ones((37, 53), np.uint8))
    assert rgb.shape == (1, 3, 40, 56) and m.shape == (1, 1, 40, 56) and hw == (37, 53)
    back = postprocess(rgb, hw)
    assert np.array_equal(back, img)
    x = np.full((1, 3, 8, 8), 0.999, np.float32)       # 254.745 -> 254 (truncation, לא 255)
    assert postprocess(x, (8, 8))[0, 0, 0] == 254


def test_ort_model_load_errors(tmp_path):
    with pytest.raises(ModelLoadError):
        OrtModel().load(ModelConfig(tmp_path / "missing.onnx"))
    lfs = tmp_path / "lfs.onnx"
    lfs.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(ModelLoadError, match="LFS"):
        OrtModel().load(ModelConfig(lfs))
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model" * 10)
    with pytest.raises(ModelLoadError):
        OrtModel().load(ModelConfig(bad))
    with pytest.raises(ModelLoadError, match="hash"):
        OrtModel().load(ModelConfig(bad, sha256="0" * 64))


def test_no_feather_is_exact_inside_mask():
    img = rnd(300, 400)
    res = Pipeline(FakeModel(), PipelineConfig(feather_px=0)).run(img, box(300, 400, 100, 140, 150, 230), 384)
    eff = res.effective_mask.astype(bool)
    assert (res.image[eff] == FILL).all() and np.array_equal(res.image[~eff], img[~eff])
