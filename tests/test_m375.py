"""M3.7.5: קנה מידה של חלון יחיד נקבע לפי החלון החתוך לתמונה (לא מבזבזים bucket על ריפוד), ו-buckets גדולים (768/1024)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.pipeline.roi import plan_roi_bbox
from test_engine import FakeModel

REPO = Path(__file__).resolve().parents[1]


def test_clipped_window_scale_uses_actual_side():
    # מסכה כמעט בגובה התמונה: החלון הרצוי (x1.5) חורג מהתמונה, והחלון החתוך הוא 5184 -> קנה המידה size/5184
    p = plan_roi_bbox((1805, 906, 3187, 5024), (5184, 3456), 1024)
    assert p.scale == pytest.approx(1024 / 5184)
    assert p.in_h == 1024 and p.in_w == round(3456 * 1024 / 5184)
    # לפני התיקון: 1024/(4118*1.5) = 0.166 ו-in_h = 859 (16% ריפוד)


def test_unclipped_window_scale_unchanged():
    p = plan_roi_bbox((1000, 1000, 1600, 1600), (4000, 4000), 384)   # side 600, win 900: בתוך התמונה
    assert p.scale == pytest.approx(384 / 900)
    assert p.in_w == p.in_h == 384


def test_native_and_small_unchanged():
    p = plan_roi_bbox((100, 100, 200, 200), (1000, 1000), 384)
    assert p.scale == 1.0


@pytest.mark.parametrize("size", [768, 1024])
def test_pipeline_single_window_at_large_bucket(size):
    img = np.random.default_rng(0).integers(0, 255, (3000, 2000, 3), dtype=np.uint8)
    mask = np.zeros((3000, 2000), np.uint8)
    mask[200:2800, 300:1700] = 255
    fm = FakeModel()
    pipe = Pipeline(fm, PipelineConfig(sizes=(256, 384, 512, 768, 1024), large_strategy="downscale"))
    res = pipe.run(img, mask, size)
    assert res.model_calls == 1 and fm.shapes == [((size, size, 3), (size, size))]
    assert res.rois[0].scale == pytest.approx(size / 3000)
    assert np.array_equal(res.image[res.effective_mask == 0], img[res.effective_mask == 0])
    # גודל שאינו ב-sizes נדחה
    with pytest.raises(Exception):
        Pipeline(FakeModel(), PipelineConfig()).run(img, mask, 1024)


def test_big_window_check_importable_and_check_size_with_fake():
    spec = importlib.util.spec_from_file_location("big_window_check", REPO / "tools" / "m3" / "big_window_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    row = m.check_size(FakeModel(), 128, 1)
    assert row["size"] == 128 and row["shape_ok"] and row["median_s"] >= 0 and "outside_mean_abs" in row


def test_bench_heavy_plan_only():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "m3" / "quality_bench.py"), str(REPO / "cases" / "paris_chair"),
                        "--plan-only", "--heavy"], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    assert "s1024-downscale" in r.stdout and "s1536-downscale" in r.stdout
