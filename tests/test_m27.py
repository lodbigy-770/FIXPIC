"""M2.7: תמונות 1/12/24MP, מסכות קטנות/גדולות/מרובות/ריקה, מודל פגום, ביטול באמצע, ו-CLI."""
import json
import shutil
import threading
import tracemalloc

import cv2
import numpy as np
import pytest

from fixpic import cli
from fixpic.engine import Job, JobScheduler, JobState
from fixpic.errors import EmptyMaskError, ModelLoadError
from fixpic.models import AutoModel, ModelConfig, ModelInfo, ModelManager
from fixpic.models.ort_model import OrtModel
from fixpic.pipeline import Pipeline, PipelineConfig, estimate_memory_mb
from test_backends import tiny  # noqa: F401  (fixture)

T = 10.0
SIZES = {"1MP": (1000, 1000), "12MP": (3000, 4000), "24MP": (4000, 6000)}


def rnd(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def rects(h, w, *rs):
    m = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in rs:
        m[y0:y1, x0:x1] = 255
    return m


@pytest.fixture(scope="module")
def ort_model(tiny):  # noqa: F811
    m = OrtModel()
    m.load(ModelConfig(tiny, threads=2))
    return m


@pytest.fixture
def models_dir(tmp_path, tiny):  # noqa: F811
    d = tmp_path / "models" / "tiny"
    d.mkdir(parents=True)
    shutil.copy(tiny, d / "w.onnx")
    (d / "model.json").write_text(json.dumps({"schema": 1, "name": "tiny", "file": "w.onnx"}), encoding="utf-8")
    return tmp_path / "models"


def check_result(img, res):
    eff = res.effective_mask.astype(bool)
    assert eff.any()
    assert np.array_equal(res.image[~eff], img[~eff])          # DoD: מחוץ למסכה זהה לבייט
    assert (res.image[eff] != img[eff]).any()                  # ובתוכה משהו מולא


# ------------------------------------------------------------- גדלי תמונה
@pytest.mark.parametrize("name", list(SIZES))
def test_image_sizes_outside_identical_and_memory_under_limit(ort_model, name):
    h, w = SIZES[name]
    img = rnd(h, w)
    mask = rects(h, w, (h // 2, h // 2 + 60, w // 2, w // 2 + 90))
    limit = estimate_memory_mb(h, w, 1600)
    pl = Pipeline(ort_model, PipelineConfig(memory_limit_mb=limit))
    before = img[::97, ::89].copy()
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = pl.run(img, mask, 384)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    check_result(img, res)
    assert np.array_equal(img[::97, ::89], before)             # הקלט לא השתנה
    assert peak <= limit - 1600                                # DoD: מתחת לתקרה (בלי תקורת המודל)
    assert res.model_calls == 1 and res.size_used == 384       # מסכה קטנה: חלון אחד בגודל bucket


# --------------------------------------------------------------- סוגי מסכה
H12, W12 = SIZES["12MP"]
MASKS = {
    "small": [(1500, 1530, 2000, 2030)],
    "thin_full_width_line": [(1500, 1503, 0, W12)],
    "large_60pct": [(600, 2400, 800, 3200)],
    "multiple_far": [(50, 120, 50, 120), (50, 120, 3800, 3900), (2900, 2980, 100, 200), (2900, 2950, 3700, 3790)],
}


@pytest.mark.parametrize("kind", list(MASKS))
@pytest.mark.parametrize("size", [256, 512])
def test_mask_kinds_at_12mp(ort_model, kind, size):
    img = rnd(H12, W12, 1)
    res = Pipeline(ort_model).run(img, rects(H12, W12, *MASKS[kind]), size)
    check_result(img, res)
    if kind == "multiple_far":
        assert res.groups == 4 and res.model_calls == 4        # קבוצות רחוקות = חלונות קטנים נפרדים


def test_empty_mask_all_sizes_no_model_call(ort_model):
    calls = []

    class Counting(OrtModel):
        def run(self, *a, **k):
            calls.append(1)
            return super().run(*a, **k)

    cm = Counting()
    cm._sess = ort_model._sess
    cm._img_name, cm._mask_name, cm._out_name = ort_model._img_name, ort_model._mask_name, ort_model._out_name
    for h, w in SIZES.values():
        with pytest.raises(EmptyMaskError):
            Pipeline(cm).run(rnd(h, w), np.zeros((h, w), np.uint8))
    assert calls == []


# ------------------------------------------------------------- מודל פגום
def corrupt_variants(tiny_path, tmp_path):
    good = tiny_path.read_bytes()
    return {"garbage": b"\x00not an onnx model" * 50, "truncated": good[: len(good) // 2], "empty": b""}


@pytest.mark.parametrize("kind", ["garbage", "truncated", "empty"])
def test_corrupt_model_file_is_model_load_error(tmp_path, tiny, kind):  # noqa: F811
    p = tmp_path / "bad.onnx"
    p.write_bytes(corrupt_variants(tiny, tmp_path)[kind])
    for m in (OrtModel(), AutoModel()):                        # גם ה-fallback האוטומטי לא מסתיר את הכשל
        with pytest.raises(ModelLoadError):
            m.load(ModelConfig(p))


def test_sha_mismatch_blocks_load(models_dir):
    mj = models_dir / "tiny" / "model.json"
    d = json.loads(mj.read_text(encoding="utf-8"))
    d["sha256"] = "0" * 64
    mj.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="hash"):
        ModelManager(models_dir).load("tiny")


# -------------------------------------------------------------- ביטול באמצע
class CountingModel:
    def __init__(self, inner):
        self.inner, self.info, self.calls = inner, inner.info, 0

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        self.calls += 1
        return self.inner.run(image, mask, cancel)


def test_cancel_midway_at_24mp_tiling_leaves_image_untouched(ort_model):
    h, w = SIZES["24MP"]
    img = rnd(h, w, 2)
    keep = img.copy()
    mask = rects(h, w, (1000, 2500, 1000, 3300))
    cm = CountingModel(ort_model)
    cfg = PipelineConfig(large_strategy="tile", max_tiles=64, mask_dilate_px=0)
    total = {}

    def on_progress(job, p):
        total["n"] = p.total
        if p.done == 2:
            job.cancel()

    with JobScheduler(Pipeline(cm, cfg)) as s:
        job = s.submit(img, mask, 512, on_progress=on_progress)
        assert job.wait(60) and job.state is JobState.CANCELLED
    assert 2 <= cm.calls < total["n"]                           # נעצר באמצע, לא בסוף
    assert np.array_equal(img, keep)                            # הקלט לא השתנה (התוצאה נזרקה)


# ------------------------------------------------------------------- CLI
@pytest.fixture
def files(tmp_path):
    d = tmp_path / "תיקייה בעברית"                              # נתיב עם עברית ורווחים (Windows)
    d.mkdir()
    img = rnd(300, 400, 5)
    cv2.imencode(".png", img[..., ::-1])[1].tofile(str(d / "תמונה.png"))
    cv2.imencode(".png", rects(300, 400, (100, 140, 150, 230)))[1].tofile(str(d / "מסכה.png"))
    return d, img


def args(d, models_dir, *extra, mask="מסכה.png", image="תמונה.png"):
    return [str(d / image), str(d / mask), "--model", "tiny", "--models-dir", str(models_dir), *extra]


def test_cli_end_to_end_hebrew_paths(files, models_dir, capsys):
    d, img = files
    out = d / "פלט.png"
    code = cli.main(args(d, models_dir, "-o", str(out), "--size", "256", "--dilate", "0", "--feather", "0", "--json"))
    assert code == 0
    info = json.loads(capsys.readouterr().out)
    assert info["model_calls"] == 1 and info["backend"] in ("openvino", "onnxruntime") and info["size"] == 256
    res = cli.read_image(out)
    outside = cli.read_mask(d / "מסכה.png") == 0
    assert np.array_equal(res[outside], img[outside]) and (res[~outside] != img[~outside]).any()
    assert not list(d.glob("*.tmp"))


def test_cli_default_output_name_and_backend_ort(files, models_dir):
    d, _ = files
    assert cli.main(args(d, models_dir, "--backend", "ort")) == 0
    assert (d / "תמונה_out.png").is_file()


@pytest.mark.parametrize("case,code", [("missing_image", 2), ("mask_size", 2), ("bad_size", 2), ("same_output", 2),
                                       ("empty_mask", 4), ("corrupt_model", 3), ("missing_model", 3)])
def test_cli_exit_codes(files, models_dir, case, code, capsys):
    d, img = files
    a = args(d, models_dir)
    if case == "missing_image":
        a = args(d, models_dir, image="אין.png")
    elif case == "mask_size":
        cv2.imencode(".png", rects(50, 50, (1, 5, 1, 5)))[1].tofile(str(d / "קטנה.png"))
        a = args(d, models_dir, mask="קטנה.png")
    elif case == "bad_size":
        a += ["--size", "300"]
    elif case == "same_output":
        a += ["-o", str(d / "תמונה.png")]
    elif case == "empty_mask":
        cv2.imencode(".png", np.zeros((300, 400), np.uint8))[1].tofile(str(d / "ריקה.png"))
        a = args(d, models_dir, mask="ריקה.png")
    elif case == "corrupt_model":
        (models_dir / "tiny" / "w.onnx").write_bytes(b"junk" * 100)
    elif case == "missing_model":
        (models_dir / "tiny" / "w.onnx").unlink()
    assert cli.main(a) == code
    assert "שגיאה" in capsys.readouterr().err
    assert not list(d.glob("*_out.png")) and not list(d.glob("*.tmp"))     # אין פלט חלקי, והקלט לא נדרס
    assert np.array_equal(cli.read_image(d / "תמונה.png"), img)


def test_cli_ctrl_c_cancels_job_exit_130(files, models_dir, monkeypatch, capsys):
    d, _ = files
    entered = threading.Event()

    class Blocking:
        info = ModelInfo(name="blocking")
        backend = "fake"

        def run(self, image, mask, cancel=None):
            entered.set()
            for _ in range(500):                                # עד 5 שניות, או עד ביטול
                if cancel.cancelled:
                    break
                threading.Event().wait(0.01)
            return image

    monkeypatch.setattr(cli, "_load_model", lambda a: Blocking())
    orig, first = Job.wait, [True]

    def wait(self, timeout=None):
        if first[0]:
            first[0] = False
            assert entered.wait(T)
            raise KeyboardInterrupt
        return orig(self, timeout)

    monkeypatch.setattr(Job, "wait", wait)
    assert cli.main(args(d, models_dir, "--dilate", "0")) == 130
    assert not (d / "תמונה_out.png").exists()
