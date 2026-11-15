"""M3.7.2: `Pipeline.plan` (תכנון בלי מודל) זהה למה ש-`run` מבצע בפועל, וכלי האבחון עובד על המקרים שבמאגר."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixpic.errors import EmptyMaskError, InvalidInputError
from fixpic.pipeline import Pipeline, PipelineConfig
from test_engine import FakeModel, rnd

REPO = Path(__file__).resolve().parents[1]


def mk(h, w, *boxes):
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 1
    return m


CASES = {
    "native": (400, 500, [(100, 100, 160, 150)], PipelineConfig()),
    "downscale": (1500, 1200, [(100, 100, 900, 800)], PipelineConfig(large_strategy="downscale")),
    "tile": (1500, 1200, [(100, 100, 900, 800)], PipelineConfig(large_strategy="tile", max_tiles=500)),
    "auto_to_tiles": (1500, 1200, [(100, 100, 900, 800)], PipelineConfig(min_scale=0.9, max_tiles=500)),
    "auto_cliff_downscale": (1500, 1200, [(100, 100, 900, 800)], PipelineConfig(min_scale=0.9, max_tiles=3)),
    "two_groups": (600, 1600, [(20, 20, 80, 80), (1400, 400, 1480, 480)], PipelineConfig()),
}


@pytest.mark.parametrize("name", list(CASES))
def test_plan_matches_run(name):
    h, w, boxes, cfg = CASES[name]
    mask = mk(h, w, *boxes)
    img = rnd(h, w, 3)
    pipe = Pipeline(FakeModel(), cfg)
    rep = pipe.plan(mask, 256, 30.0)
    res = pipe.run(img, mask, 256, brush_px=30.0)
    assert len(rep.groups) == res.groups
    assert rep.runs == res.model_calls == len(res.rois)
    assert rep.dilate_px == res.dilate_px and rep.mask_px == int(res.effective_mask.sum())
    for g in rep.groups:
        assert g.kind in ("native", "downscale", "tile") and g.windows >= 1
    if name in ("downscale", "auto_cliff_downscale"):
        assert all(g.kind == "downscale" and g.scale < 1 for g in rep.groups)
        assert all(r.scale == g.scale for r, g in zip(res.rois, rep.groups))
    if name in ("tile", "auto_to_tiles"):
        assert all(g.kind == "tile" and g.scale == 1.0 for g in rep.groups) and rep.runs > 1
    if name == "native":
        assert rep.groups[0].kind == "native" and rep.runs == 1


def test_plan_validates_like_run():
    pipe = Pipeline(FakeModel(), PipelineConfig())
    with pytest.raises(EmptyMaskError):
        pipe.plan(np.zeros((50, 50), np.uint8))
    with pytest.raises(InvalidInputError):
        pipe.plan(mk(50, 50, (1, 1, 5, 5)), 100)
    with pytest.raises(InvalidInputError):
        pipe.plan(np.zeros((5, 5, 3), np.uint8))


def test_large_region_has_no_cliff_anymore():
    """M3.7.6, מקרה boy_large: ב-384 אריחים מוקטנים מעט (ולא חלון יחיד פי 16); ב-512 אריחים מקוריים."""
    mask = mk(5184, 3456, (1785, 886, 3208, 5045))
    pipe = Pipeline(FakeModel(), PipelineConfig())
    r384, r512 = pipe.plan(mask, 384, 100.0), pipe.plan(mask, 512, 100.0)
    g = r384.groups[0]
    assert g.kind == "tile" and 0.5 < g.scale < 1.0 and g.windows <= PipelineConfig().max_tiles
    assert r512.groups[0].kind == "tile" and r512.min_scale == 1.0


def test_scale_degrades_gradually_with_region_size():
    pipe = Pipeline(FakeModel(), PipelineConfig())
    scales = []
    for side in (300, 800, 1500, 2500, 4000, 6000, 9000):
        rep = pipe.plan(mk(9500, 9500, (100, 100, 100 + side, 100 + side)), 384, 60.0)
        assert rep.runs <= PipelineConfig().max_tiles or rep.groups[0].kind != "tile"
        scales.append(rep.min_scale)
    assert all(a >= b - 1e-9 for a, b in zip(scales, scales[1:]))                      # לא עולה כשהאזור גדל
    assert all(a - b < 0.5 for a, b in zip(scales, scales[1:]))                        # ואין קפיצה (מצוק)
    assert scales[-1] > 0.15


def test_downscaled_tiles_fill_mask_only_and_progress_matches():
    h, w = 1800, 1400
    mask = mk(h, w, (100, 100, 1300, 1700))
    img = rnd(h, w, 9)
    events = []
    pipe = Pipeline(FakeModel(), PipelineConfig(max_tiles=6, feather_px=0))
    rep = pipe.plan(mask, 256, 40.0)
    assert rep.groups[0].kind == "tile" and rep.groups[0].scale < 1.0 and rep.runs <= 6
    res = pipe.run(img, mask, 256, brush_px=40.0, progress=lambda p: events.append(p))
    eff = res.effective_mask > 0
    assert np.array_equal(res.image[~eff], img[~eff])                                   # מחוץ למסכה: זהה לבייט
    assert (res.image[eff] != img[eff]).any(axis=1).mean() > 0.95                     # כל המסכה מולאה (FakeModel מוסיף 7)
    assert res.model_calls == rep.runs and events[-1].done == events[-1].total == rep.runs
    assert all(r.scale == rep.groups[0].scale for r in res.rois)


@pytest.mark.skipif(not (REPO / "cases").exists(), reason="cases/ לא קיים")
def test_diag_tool_runs_on_checked_in_cases():
    out = subprocess.run([sys.executable, str(REPO / "tools/m3/large_region_diag.py"), str(REPO / "cases/boy_large"),
                          str(REPO / "cases/paris_chair"), "--sizes", "384,512"], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert "boy_large" in out.stdout and "paris_chair" in out.stdout and "downscale" in out.stdout
