"""M2.9: טיפול בשגיאות לפי ARCHITECTURE §10 (שורה לשורה) ובדיקת DoD של M2: אין קריסות,
הקלט אינו משתנה בכשל, ה-Worker ממשיך לעבוד אחרי כל כשל."""
import contextlib
import json
import logging
import shutil
import threading

import numpy as np
import pytest

from fixpic.engine import JobScheduler, JobState
from fixpic.errors import (CancelledError, EmptyMaskError, FixPicError, InferenceError, InvalidInputError,
                           ModelLoadError, OutOfMemoryError)
from fixpic.models import AutoModel, ModelConfig, ModelManager
from fixpic.pipeline import Pipeline, PipelineConfig
from test_backends import tiny  # noqa: F401  (fixture)
from test_engine import FILL, FakeModel, rnd
from test_m24 import blocks  # noqa: F401

T = 10.0


@contextlib.contextmanager
def sched(pipeline):
    s = JobScheduler(pipeline).start(warmup=False)
    try:
        yield s
    finally:
        s.shutdown()


def box(h, w, y0=100, y1=140, x0=100, x1=140):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


class Raises(FakeModel):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def run(self, image, mask, cancel=None):
        raise self.exc


class Prim(FakeModel):
    """backend ראשי שנכשל בהרצה (מדמה OpenVINO)."""
    def run(self, image, mask, cancel=None):
        raise InferenceError("ov נכשל (מדומה)")


class BadLoad(FakeModel):
    def load(self, cfg):
        raise ModelLoadError("טעינה נכשלה (מדומה)")


@pytest.fixture
def no_file_check(monkeypatch):
    monkeypatch.setattr("fixpic.models.auto.check_model_file", lambda cfg: None)


# ------------------------------------------------- שורה 1: מודל חסר / hash שגוי
def test_missing_model_blocks_inpaint_and_is_logged(tmp_path, caplog):
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.json").write_text(json.dumps({"schema": 1, "name": "m", "file": "nope.onnx"}), encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="fixpic.models"), pytest.raises(ModelLoadError):
        ModelManager(tmp_path).load("m")
    assert any("נכשלה" in r.message for r in caplog.records)


def test_hash_mismatch_blocks_inpaint_logged_and_leaves_no_active_backend(tiny, tmp_path, caplog):  # noqa: F811
    m = AutoModel()
    with caplog.at_level(logging.ERROR, logger="fixpic.models"), pytest.raises(ModelLoadError, match="hash"):
        m.load(ModelConfig(tiny, sha256="0" * 64))
    assert any("נחסמה" in r.message for r in caplog.records)
    assert m.backend == ""
    img, mask = rnd(200, 200), box(200, 200)
    orig = img.copy()
    with pytest.raises(InferenceError):                       # Inpaint חסום: המודל לא נטען
        Pipeline(m).run(img, mask)
    assert np.array_equal(img, orig)


def test_hash_mismatch_after_good_load_drops_active_backend(tiny):  # noqa: F811
    m = AutoModel()
    m.load(ModelConfig(tiny))
    assert m.backend
    with pytest.raises(ModelLoadError):
        m.load(ModelConfig(tiny, sha256="0" * 64))
    assert m.backend == ""


# ------------------------------------------------- שורה 2: OOM
def test_persistent_oom_fails_cleanly_input_unchanged_and_worker_alive():
    class OomThenOk(FakeModel):
        bad = True

        def run(self, image, mask, cancel=None):
            if self.bad:
                raise MemoryError()
            return super().run(image, mask, cancel)

    model = OomThenOk()
    img, mask = rnd(300, 300), box(300, 300)
    orig = img.copy()
    with sched(Pipeline(model)) as s:
        j = s.submit(img, mask)
        j.wait(T)
        assert j.state is JobState.FAILED
        with pytest.raises(OutOfMemoryError):
            j.result(0)
        assert np.array_equal(img, orig)
        model.bad = False
        j2 = s.submit(img, mask)
        assert j2.wait(T) and j2.state is JobState.DONE      # אותו Worker ממשיך


# ------------------------------------------------- שורה 3: חריגת backend
@pytest.mark.parametrize("exc", [ValueError("x"), RuntimeError("ort"), KeyError("k"), ZeroDivisionError()])
def test_unexpected_backend_exception_becomes_defined_error_in_pipeline(exc):
    img, mask = rnd(300, 300), box(300, 300)
    orig = img.copy()
    with pytest.raises(InferenceError) as ei:
        Pipeline(Raises(exc)).run(img, mask)
    assert isinstance(ei.value, FixPicError) and ei.value.__cause__ is exc
    assert np.array_equal(img, orig)


def test_memoryerror_outside_model_call_becomes_oom(monkeypatch):
    import fixpic.pipeline.pipeline as pp

    def boom(*a, **k):
        raise MemoryError()
    monkeypatch.setattr(pp, "estimate_brush_px", boom)
    with pytest.raises(OutOfMemoryError):
        Pipeline(FakeModel()).run(rnd(300, 300), box(300, 300))


def test_failed_backend_and_failed_fallback_leave_defined_state(no_file_check):
    a = AutoModel(primary=Prim(), fallback=BadLoad())
    a.load(ModelConfig("x"))
    z, zm = np.zeros((256, 256, 3), np.uint8), np.zeros((256, 256), np.uint8)
    with pytest.raises(ModelLoadError):
        a.run(z, zm)
    assert a.backend == ""
    for _ in range(2):                                        # אין לולאת ניסיונות: כשל קבוע ומוגדר
        with pytest.raises(InferenceError, match="לא נטען"):
            a.run(z, zm)


# ------------------------------------------------- שורה 4: מסכה ריקה
@pytest.mark.parametrize("mask", [np.zeros((300, 300), np.uint8), np.zeros((300, 300), bool),
                                  np.zeros((300, 300), np.float32)])
def test_empty_mask_does_not_run_model(mask):
    model = FakeModel()
    with pytest.raises(EmptyMaskError):
        Pipeline(model).run(rnd(300, 300), mask)
    assert model.shapes == []


# ------------------------------------------------- שורה 5: ביטול
def test_cancel_mid_run_leaves_input_unchanged():
    started, release = threading.Event(), threading.Event()

    class Slow(FakeModel):
        def run(self, image, mask, cancel=None):
            started.set()
            release.wait(T)
            return super().run(image, mask, cancel)

    img, mask = rnd(300, 300), box(300, 300)
    orig = img.copy()
    with sched(Pipeline(Slow())) as s:
        j = s.submit(img, mask)
        assert started.wait(T)
        j.cancel()
        release.set()
        assert j.wait(T) and j.state is JobState.CANCELLED
        with pytest.raises(CancelledError):
            j.result(0)
    assert np.array_equal(img, orig)


# ------------------------------------------------- שורה 6: קלט לא תקין
@pytest.mark.parametrize("img,mask", [
    (None, box(300, 300)), (rnd(300, 300).tolist(), box(300, 300)),           # לא ndarray
    (rnd(300, 300).astype(np.float32), box(300, 300)),                        # dtype שגוי
    (rnd(300, 300)[..., :2], box(300, 300)),                                   # ערוצים
    (np.zeros((0, 50, 3), np.uint8), np.zeros((0, 50), np.uint8)),            # ריקה
    (rnd(300, 300), None), (rnd(300, 300), box(200, 300)),                    # מסכה
    (rnd(300, 300), np.stack([box(300, 300)] * 3, -1)),
])
def test_invalid_input_is_defined_error(img, mask):
    with pytest.raises(InvalidInputError):
        Pipeline(FakeModel()).run(img, mask)


def test_invalid_size_is_defined_error():
    with pytest.raises(InvalidInputError):
        Pipeline(FakeModel()).run(rnd(300, 300), box(300, 300), size=300)


# ------------------------------------------------- העיקרון: כשל אינו משנה את התמונה
def test_failure_midway_returns_nothing_and_input_untouched_multi_group():
    calls = []

    class FailSecond(FakeModel):
        def run(self, image, mask, cancel=None):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("כשל בחלון השני")
            return super().run(image, mask, cancel)

    img = rnd(1200, 1200)
    mask = np.zeros((1200, 1200), np.uint8)
    for y, x in [(50, 50), (50, 1000), (1000, 50), (1000, 1000)]:
        mask[y:y + 60, x:x + 60] = 255
    orig = img.copy()
    with pytest.raises(InferenceError):
        Pipeline(FailSecond()).run(img, mask)
    assert len(calls) == 2 and np.array_equal(img, orig)


# ------------------------------------------------- DoD: רצף כשלים ועבודות תקינות על Worker יחיד
def test_dod_failure_sequence_never_kills_worker_and_outputs_stay_exact(tiny, tmp_path):  # noqa: F811
    d = tmp_path / "models" / "tiny"
    d.mkdir(parents=True)
    shutil.copy(tiny, d / "w.onnx")
    (d / "model.json").write_text(json.dumps({"schema": 1, "name": "tiny", "file": "w.onnx"}), encoding="utf-8")
    model = ModelManager(tmp_path / "models").load("tiny", threads=2)
    good = [(rnd(1000, 1000, 1), box(1000, 1000, 400, 520, 300, 700)),
            (rnd(3000, 4000, 2), box(3000, 4000, 1000, 1100, 1500, 2500))]
    bad = [(rnd(300, 300), np.zeros((300, 300), np.uint8), EmptyMaskError),
           (rnd(300, 300).astype(np.float32), box(300, 300), InvalidInputError),
           (rnd(300, 300), box(300, 200), InvalidInputError)]
    with sched(Pipeline(model, PipelineConfig(memory_limit_mb=4000))) as s:
        for i, (img, mask) in enumerate(good):
            for b_img, b_mask, err in bad:                       # הכשל חוזר דרך העבודה, לא דרך submit
                bj = s.submit(b_img, b_mask)
                assert bj.wait(T) and bj.state is JobState.FAILED
                with pytest.raises(err):
                    bj.result(0)
            j = s.submit(img, mask)
            assert j.wait(120) and j.state is JobState.DONE, f"good {i}"
            eff = j.result(0).effective_mask.astype(bool)
            assert np.array_equal(j.result(0).image[~eff], img[~eff])           # מחוץ למסכה זהה
            assert (j.result(0).image[eff] != img[eff]).any()
