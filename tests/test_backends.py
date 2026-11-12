import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper as h

from fixpic.errors import InferenceError, ModelLoadError
from fixpic.models import AutoModel, ModelConfig
from fixpic.models.ort_model import OrtModel
from fixpic.models.ov_model import OvModel
from fixpic.pipeline import Pipeline


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """מודל ONNX קטן עם אותו חוזה כמו Big LaMa: rgb/mask דינמיים -> inpainted.
    out = rgb*(1-mask) + mask*(1 - rgb) (פלט תלוי בקלט, כדי שפריטיות תהיה משמעותית)."""
    rgb = h.make_tensor_value_info("rgb", T.FLOAT, [1, 3, "8*height", "8*width"])
    mask = h.make_tensor_value_info("mask", T.FLOAT, [1, 1, "8*height", "8*width"])
    out = h.make_tensor_value_info("inpainted", T.FLOAT, [1, 3, "8*height", "8*width"])
    one = h.make_tensor("one", T.FLOAT, [], [1.0])
    nodes = [h.make_node("Sub", ["one", "mask"], ["inv"]), h.make_node("Mul", ["rgb", "inv"], ["keep"]),
             h.make_node("Sub", ["one", "rgb"], ["flip"]), h.make_node("Mul", ["flip", "mask"], ["fill"]),
             h.make_node("Add", ["keep", "fill"], ["inpainted"])]
    g = h.make_graph(nodes, "t", [rgb, mask], [out], [one])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    p = tmp_path_factory.mktemp("m") / "tiny.onnx"
    onnx.save(m, p)
    return p


def sample(h_=300, w=400):
    img = np.random.default_rng(3).integers(0, 256, (h_, w, 3), dtype=np.uint8)
    mask = np.zeros((h_, w), np.uint8)
    mask[100:140, 150:230] = 255
    return img, mask


def test_ov_ort_parity_through_pipeline(tiny):
    cfg = ModelConfig(tiny, threads=3)
    ort, ov = OrtModel(), OvModel()
    ort.load(cfg); ov.load(cfg)
    img, mask = sample()
    a = Pipeline(ort).run(img, mask, 256).image.astype(int)
    b = Pipeline(ov).run(img, mask, 256).image.astype(int)
    assert np.abs(a - b).max() <= 3          # ADR-011: עד 3 ברמות (סקאלה 0-255)


def test_ov_shape_cache_and_warmup(tiny):
    ov = OvModel(); ov.load(ModelConfig(tiny))
    pl = Pipeline(ov)
    pl.warmup()
    assert set(ov._compiled) == {(256, 256), (384, 384), (512, 512)}


def test_auto_prefers_openvino(tiny):
    m = AutoModel(); m.load(ModelConfig(tiny))
    assert m.backend == "openvino" and m.fallback_reason is None
    img, mask = sample()
    assert Pipeline(m).run(img, mask).image.shape == img.shape


def test_auto_falls_back_when_openvino_missing(tiny):
    m = AutoModel(primary=None)
    m._primary = None                       # כמו import שנכשל
    m.load(ModelConfig(tiny))
    assert m.backend == "onnxruntime" and m.fallback_reason
    img, mask = sample()
    assert Pipeline(m).run(img, mask).image.shape == img.shape


class _BrokenCompile(OvModel):
    def _get(self, h, w):
        raise ModelLoadError("compile נכשל (מדומה)")


class _BrokenRun(OvModel):
    def run(self, image, mask, cancel=None):
        raise InferenceError("כשל בהרצה (מדומה)")


@pytest.mark.parametrize("broken", [_BrokenCompile, _BrokenRun])
def test_auto_runtime_failure_switches_to_ort(tiny, broken):
    m = AutoModel(primary=broken())
    m.load(ModelConfig(tiny))
    assert m.backend == "openvino"
    img, mask = sample()
    res = Pipeline(m).run(img, mask)            # בלי חריגה
    assert m.backend == "onnxruntime" and m.fallback_reason
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])


def test_auto_missing_model_is_direct_error(tmp_path):
    with pytest.raises(ModelLoadError):
        AutoModel().load(ModelConfig(tmp_path / "x.onnx"))
