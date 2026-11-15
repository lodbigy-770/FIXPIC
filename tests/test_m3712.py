"""M3.7.12: ציר `--dilates` ב-`tools/m3/quality_bench.py` (הרחבת מסכה לפי קנה המידה של המודל), על מודל מדומה."""
from __future__ import annotations

import subprocess
import sys

import pytest

from fixpic.pipeline import Pipeline, PipelineConfig
from test_engine import FakeModel
from test_m374 import REPO, case, mk_mask, qb  # noqa: F401  (case = fixture)


def test_parse_dilate_normalizes_and_rejects():
    assert qb.parse_dilate("AUTO") == "auto"
    assert qb.parse_dilate(" 060 ") == "60"
    assert qb.parse_dilate("M08") == "m8"
    for bad in ("", "x", "m", "-5", "m-1", "1.5"):
        with pytest.raises(ValueError):
            qb.parse_dilate(bad)


def test_resolve_dilate_auto_fixed_and_cap():
    mask = mk_mask(1500, 1200, (100, 100, 900, 800))
    pc = PipelineConfig()
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "auto") is None
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "0") == 0
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "60") == 60
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "99999") == qb.MAX_DILATE_PX


def test_resolve_dilate_model_px_scales_with_downscale():
    mask = mk_mask(1500, 1200, (100, 100, 900, 800))
    pc = PipelineConfig(coarse_fine=False, large_strategy="downscale")
    r = qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "m8")
    rep = Pipeline(FakeModel(), PipelineConfig(coarse_fine=False, large_strategy="downscale", mask_dilate_px=r)).plan(mask, 384)
    assert rep.min_scale < 0.6 and r > 8
    assert 8 <= rep.dilate_px * rep.min_scale < 9.5        # לפחות 8 פיקסלים בקלט המודל, בלי ניפוח
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "m0") == 0


def test_resolve_dilate_model_px_native_scale_equals_n():
    mask = mk_mask(400, 500, (100, 100, 160, 150))          # נכנס בלי הקטנה
    pc = PipelineConfig(coarse_fine=False)
    assert Pipeline(FakeModel(), pc).plan(mask, 384).min_scale == 1.0
    assert qb.resolve_dilate(FakeModel(), pc, mask, 384, None, "m8") == 8


def test_build_grid_dilate_ids_and_default_unchanged():
    base = qb.build_grid([384], ["downscale"], [0.5], [3], [64])
    assert [c["id"] for c in base] == ["s384-downscale-f3-o64-b32"] and base[0]["dilate"] == "auto"
    g = qb.build_grid([384], ["downscale"], [0.5], [3], [64], dilates=["auto", "20", "M8"])
    assert [c["id"] for c in g] == ["s384-downscale-f3-o64-b32", "s384-downscale-f3-o64-b32-d20", "s384-downscale-f3-o64-b32-dm8"]


def test_plan_signature_distinguishes_dilation(case):
    from fixpic.casefile import load_case
    mask = load_case(case)["mask"]
    a = Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=5)).plan(mask, 384)
    b = Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=60)).plan(mask, 384)
    assert qb.plan_signature(a, 64) != qb.plan_signature(b, 64)


def test_run_bench_dilate_axis_records_px_and_grows_mask(case, tmp_path):
    cfgs = qb.build_grid([384], ["downscale"], [0.5], [3], [64], dilates=["0", "40", "m8"])
    data = qb.run_bench(FakeModel(), case, cfgs, tmp_path / "o", log=lambda *_: None)
    rows = {r["id"].rsplit("-", 1)[-1]: r for r in data["rows"] if not r.get("reference")}
    assert set(rows) == {"d0", "d40", "dm8"}
    assert rows["d0"]["dilate_px"] == 0 and rows["d40"]["dilate_px"] == 40 and rows["dm8"]["dilate_px"] > 8
    assert all("error" not in r and r["seconds"] >= 0 for r in rows.values())


def test_cli_plan_only_dilates(case):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "m3" / "quality_bench.py"), str(case), "--plan-only", "--sizes", "384",
                        "--strategies", "downscale", "--dilates", "auto,60,m8"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    assert "-d60" in r.stdout and "-dm8" in r.stdout and "הרחבה" in r.stdout


def test_cli_rejects_bad_dilate(case):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "m3" / "quality_bench.py"), str(case), "--plan-only", "--dilates", "zz"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode != 0
