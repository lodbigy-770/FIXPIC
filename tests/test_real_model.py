"""רץ רק כשקובץ המודל האמיתי קיים (לא LFS pointer). אצל המשתמש: python -m pytest tests/test_real_model.py"""
from pathlib import Path

import numpy as np
import pytest

from fixpic.models import ModelConfig
from fixpic.models.ort_model import OrtModel
from fixpic.pipeline import Pipeline

MODEL = Path(__file__).resolve().parents[1] / "models" / "big-lama" / "inpainter.onnx"
pytestmark = pytest.mark.skipif(not MODEL.is_file() or MODEL.stat().st_size < 1_000_000,
                                reason="המודל האמיתי לא זמין (LFS pointer)")


@pytest.mark.parametrize("size", [256, 384])
def test_real_model_outside_identical(size):
    m = OrtModel()
    m.load(ModelConfig(MODEL, threads=3))
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    mask = np.zeros((600, 800), np.uint8)
    mask[250:300, 350:420] = 255
    res = Pipeline(m).run(img, mask, size)
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])
    assert (res.image[eff] != img[eff]).any()


def test_real_model_ov_vs_ort_parity():
    pytest.importorskip("openvino")
    from fixpic.models.ov_model import OvModel
    cfg = ModelConfig(MODEL, threads=3)
    ort, ov = OrtModel(), OvModel()
    ort.load(cfg)
    ov.load(cfg)
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (384, 384, 3), dtype=np.uint8)
    mask = np.zeros((384, 384), np.uint8)
    mask[150:230, 120:260] = 1
    d = np.abs(ort.run(img, mask).astype(int) - ov.run(img, mask).astype(int))[mask > 0]
    assert d.max() <= 3     # ADR-011
