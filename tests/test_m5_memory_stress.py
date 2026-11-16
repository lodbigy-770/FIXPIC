"""M5: מבחן תמונות גדולות ועומס זיכרון (`tools/m5/memory_stress.py`), על גדלים קטנים בלבד (המדידה האמיתית אצל המשתמש)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_stress", REPO / "tools" / "m5" / "memory_stress.py")
ms = importlib.util.module_from_spec(spec)
sys.modules["memory_stress"] = ms
spec.loader.exec_module(ms)


def test_dims_for_mp_aspect_and_size():
    w, h = ms.dims_for_mp(12)
    assert abs(w / h - 1.5) < 0.01 and abs(w * h / 1e6 - 12) < 0.1
    assert ms.dims_for_mp(0.0001)[1] >= 1


def test_needed_mb_grows_and_exceeds_pipeline_estimate():
    from fixpic.pipeline.pipeline import estimate_memory_mb
    a, b = ms.needed_mb(2000, 1000), ms.needed_mb(4000, 2000)
    assert b > a > estimate_memory_mb(1000, 2000)
    assert ms.needed_mb(2000, 1000, model_overhead_mb=500) == a + 500


def test_peak_rss_positive_and_monotonic():
    a = ms.peak_rss_mb()
    buf = bytearray(30 * 2**20)
    buf[::4096] = b"\1" * len(buf[::4096])
    assert a > 0 and ms.peak_rss_mb() >= a
    assert ms.physical_memory_mb() > 100


def test_gen_image_makes_requested_jpeg(tmp_path):
    p = tmp_path / "g.jpg"
    ms.gen_image(p, 700, 400)
    with Image.open(p) as im:
        assert im.size == (700, 400) and im.format == "JPEG"


def test_measure_child_ok_and_fields(tmp_path):
    p = tmp_path / "g.jpg"
    ms.gen_image(p, 800, 600)
    r = ms.measure_child(p, 0.2, 384, None, 1, tmp_path)
    assert r["ok"] is True and (r["w"], r["h"]) == (800, 600) and r["mask_pct"] == 20.0
    assert r["baseline_mb"] > 0 and r["load_peak_mb"] <= r["run_peak_mb"] <= r["save_peak_mb"]
    assert (tmp_path / "out.jpg").is_file() and r["out_mb"] >= 0


def test_measure_child_reports_failure_instead_of_raising(tmp_path):
    r = ms.measure_child(tmp_path / "missing.jpg", 0.1, 384, None, 1, tmp_path)
    assert r["ok"] is False and "error" in r


def test_run_one_skips_over_budget_without_creating_image(tmp_path):
    r = ms.run_one(100, 0.1, 384, None, 1, budget_mb=100, tmp=tmp_path)
    assert r["ok"] is None and "חורגת" in r["skipped"] and not list(tmp_path.iterdir())


def test_main_end_to_end_small(tmp_path):
    out_json, out_md = tmp_path / "o.json", tmp_path / "o.md"
    rc = ms.main(["--mps", "0.4", "50", "--mask-fracs", "0.1", "--budget-mb", "500", "--tmp", str(tmp_path / "w"),
                  "--json", str(out_json), "--md", str(out_md)])
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert [r["ok"] for r in data["rows"]] == [True, None] and data["info"]["model"] == "synthetic"
    md = out_md.read_text(encoding="utf-8")
    assert "| MP |" in md and "דולג" in md and "ok" in md
