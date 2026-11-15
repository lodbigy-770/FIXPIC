"""M3.7.4: ערכת ההשוואה (`tools/m3/quality_bench.py`) ו-`Pipeline.tile_seams`, על מודל מדומה."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from fixpic.casefile import save_case
from fixpic.pipeline import Pipeline, PipelineConfig
from test_engine import FakeModel

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quality_bench", REPO / "tools" / "m3" / "quality_bench.py")
qb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qb)


def noise_img(h, w, seed=0):
    r = np.random.default_rng(seed)
    n = cv2.GaussianBlur(r.normal(0, 8, (h, w, 3)).astype(np.float32), (0, 0), 1.0) * 2
    return np.clip(120 + n, 0, 255).astype(np.uint8)


def mk_mask(h, w, box):
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return m


@pytest.fixture()
def case(tmp_path):
    h, w = 1500, 1200
    img, mask = noise_img(h, w), mk_mask(h, w, (100, 100, 900, 800))
    ref = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return save_case(tmp_path / "c1", mask=mask, original_rgb=img, reference_rgb=ref, params={"brush_px": 40})


# ---------------------------------------------------------------- tile_seams

def test_tile_seams_lines_between_tiles():
    mask = mk_mask(1500, 1200, (100, 100, 900, 800)) > 0
    pipe = Pipeline(FakeModel(), PipelineConfig(coarse_fine=False, large_strategy="tile", max_tiles=500))
    lines = pipe.tile_seams(mask)
    assert lines and {k for k, *_ in lines} == {"v", "h"}
    plan = pipe.plan(mask)
    assert plan.groups[0].kind == "tile"
    for kind, c, a, b in lines:
        assert b - a >= 16
        x0, y0, x1, y1 = plan.groups[0].box
        assert (x0 < c < x1) if kind == "v" else (y0 < c < y1)


def test_tile_seams_empty_without_tiles():
    mask = mk_mask(400, 500, (100, 100, 160, 150)) > 0
    assert Pipeline(FakeModel(), PipelineConfig(coarse_fine=False)).tile_seams(mask) == []


# ---------------------------------------------------------------- רשת

def test_build_grid_dedupes_and_min_scale_only_for_auto():
    g = qb.build_grid([384], ["downscale", "auto", "tile"], [0.5, 0.25], [3], [64])
    ids = [c["id"] for c in g]
    assert ids == ["s384-downscale-f3-o64-b32", "s384-auto-ms0.5-f3-o64-b32", "s384-auto-ms0.25-f3-o64-b32", "s384-tile-f3-o64-b32"]


def test_plan_signature_ignores_overlap_without_tiles():
    from fixpic.models.base import ModelInfo

    class Stub:
        info = ModelInfo(name="s", multiple=8, min_size=64, max_size=4096)

    mask = mk_mask(400, 500, (100, 100, 160, 150))
    pipe = Pipeline(Stub(), PipelineConfig(coarse_fine=False))
    rep = pipe.plan(mask, 384)
    assert qb.plan_signature(rep, 64) == qb.plan_signature(rep, 128)


# ---------------------------------------------------------------- הרצה מלאה

def test_run_bench_outputs(case, tmp_path):
    cfgs = qb.build_grid([384], ["downscale", "tile"], [0.5], [3, 8], [64])
    data = qb.run_bench(FakeModel(), case, cfgs, tmp_path / "out", log=lambda *_: None)
    out = tmp_path / "out" / "c1"
    for f in ("results.json", "results.csv", "summary.txt", "sheet.jpg"):
        assert (out / f).is_file()
    done = [r for r in data["rows"] if not r.get("reference")]
    assert len(done) == 4 and all("texture_ratio" in r and r["seconds"] >= 0 for r in done)
    tile = [r for r in done if r["kinds"] == "tile"]
    down = [r for r in done if r["kinds"] == "downscale"]
    assert tile and down and all(r["seam_ratio"] is not None and r["seam_lines"] > 0 for r in tile)
    assert all(r["seam_ratio"] is None for r in down)
    assert all(r["planned_runs"] > 1 for r in tile) and all(r["planned_runs"] == 1 for r in down)
    # הייחוס קטן מהמקור: נמדד גם בגודל הייחוס, ושורת reference קיימת
    assert all("ref_texture_ratio" in r for r in done)
    assert data["rows"][0]["reference"] is True and "ref_texture_ratio" in data["rows"][0]
    assert len(list((out / "crops").glob("*.png"))) == 4
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["case"] == "c1"
    sheet = cv2.imread(str(out / "sheet.jpg"))
    assert sheet is not None and sheet.shape[1] == 4 * 420
    assert "reference" in (out / "summary.txt").read_text(encoding="utf-8")


def test_run_bench_skips_duplicates_and_over_max_runs(case, tmp_path):
    cfgs = qb.build_grid([384], ["auto", "tile"], [0.5, 0.25], [3], [64])
    data = qb.run_bench(FakeModel(), case, cfgs, tmp_path / "o", max_runs=2, log=lambda *_: None)
    reasons = " ".join(s["reason"] for s in data["skipped"])
    assert "max-runs" in reasons or "מעל" in reasons
    cfgs2 = qb.build_grid([384], ["auto", "tile"], [0.5, 0.25], [3], [64])
    data2 = qb.run_bench(FakeModel(), case, cfgs2, tmp_path / "o2", max_runs=500, log=lambda *_: None)
    assert any("זהה" in s["reason"] for s in data2["skipped"])
    assert len([r for r in data2["rows"] if not r.get("reference")]) < len(cfgs2)


def test_run_bench_records_failure_and_continues(case, tmp_path):
    from fixpic.errors import InferenceError

    class Boom(FakeModel):
        def run(self, image, mask, cancel=None):
            raise InferenceError("boom")

    data = qb.run_bench(Boom(), case, qb.build_grid([384], ["downscale"], [0.5], [3], [64]), tmp_path / "o",
                        log=lambda *_: None)
    assert "InferenceError" in data["rows"][-1]["error"]


def test_cli_plan_only(case):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "m3" / "quality_bench.py"), str(case), "--plan-only", "--quick"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    assert "s384-tile-f3-o64-b0" in r.stdout


def test_blend_axis_runs_both_and_dedupes_without_tiles(case, tmp_path):
    cfgs = qb.build_grid([384], ["downscale", "tile"], [0.5], [3], [64], [0, 32])
    assert len(cfgs) == 4
    data = qb.run_bench(FakeModel(), case, cfgs, tmp_path / "o", log=lambda *_: None)
    done = [r["id"] for r in data["rows"] if not r.get("reference")]
    assert "s384-tile-f3-o64-b0" in done and "s384-tile-f3-o64-b32" in done
    assert sum("downscale" in i for i in done) == 1 and any("זהה" in s["reason"] for s in data["skipped"])


def test_analyze_bench_roundtrip(case, tmp_path):
    spec2 = importlib.util.spec_from_file_location("analyze_bench", REPO / "tools" / "m3" / "analyze_bench.py")
    ab = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ab)
    cfgs = qb.build_grid([384], ["downscale", "tile"], [0.5], [3], [64], [32])
    qb.run_bench(FakeModel(), case, cfgs, tmp_path / "o", log=lambda *_: None)
    lines = ab.analyze_case(tmp_path / "o" / "c1", case.parent)
    text = "\n".join(lines)
    assert "reference" in text and "s384-tile-f3-o64-b32" in text and "s384-downscale-f3-o64-b32" in text
    ref = {"texture_ratio": 0.45}
    good = dict(texture_ratio=0.44, stripe_score=5.0, seam_ratio=None)
    assert ab.verdict(good, 5.0, ref) == "תקין"                       # הייחוס עצמו נמוך: יחסית אליו תקין
    assert "מרוח" in ab.verdict(dict(good, texture_ratio=0.2), 5.0, None)
    assert "seam" in ab.verdict(dict(good, texture_ratio=0.9, seam_ratio=1.5), 5.0, None)
    assert "stripe" in ab.verdict(dict(good, texture_ratio=0.9, stripe_score=70.0), 5.0, None)
    assert ab.verdict(dict(good, texture_ratio=0.9, stripe_score=11.0), 3.0, None) == "תקין"   # מתחת לרצפה 12
