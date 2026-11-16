"""M3.7.11: ערכת הקבלה `tools/m3/acceptance.py` (מודל מדומה; הקבלה האמיתית על Big LaMa אצל המשתמש)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_engine import FakeModel, rnd

from fixpic.casefile import save_case
from fixpic.pipeline import PipelineConfig

_spec = importlib.util.spec_from_file_location("acceptance", Path(__file__).resolve().parents[1] / "tools" / "m3" / "acceptance.py")
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)


def _case(tmp_path, name, h, w, ref_hw=None, seed=0):
    m = np.zeros((h, w), np.uint8)
    m[h // 4:3 * h // 4, w // 3:2 * w // 3] = 255
    ref = rnd(*ref_hw, seed=seed + 1) if ref_hw else None
    return save_case(tmp_path / name, mask=m, original_rgb=rnd(h, w, seed=seed), reference_rgb=ref)


def test_run_all_writes_sheet_metrics_and_accept_md(tmp_path):
    c1 = _case(tmp_path, "with_ref", 400, 300, ref_hw=(200, 150))
    c2 = _case(tmp_path, "no_ref", 300, 400, seed=5)
    out = tmp_path / "out"
    logs: list[str] = []
    rows = acc.run_all(FakeModel(), [c1, c2], out, size=256, log=logs.append)
    assert [r["case"] for r in rows] == ["with_ref", "no_ref"] and all("error" not in r for r in rows)
    for n in ("with_ref", "no_ref"):
        assert (out / n / "sheet.jpg").stat().st_size > 0 and (out / n / "ours.jpg").exists() and (out / n / "result.json").exists()
    assert (out / "with_ref" / "ours_refsize.png").exists() and not (out / "no_ref" / "ours_refsize.png").exists()
    assert "ref_texture_ratio" in rows[0] and "reference_texture_ratio" in rows[0] and "ref_texture_ratio" not in rows[1]
    md = (out / "ACCEPT.md").read_text(encoding="utf-8")
    assert "with_ref" in md and "no_ref" in md and "ממצא בעין" in md and "coarse_size=768" in md
    assert len(json.loads((out / "results.json").read_text(encoding="utf-8"))) == 2 and len(logs) == 2


def test_failure_is_reported_not_raised(tmp_path):
    c = _case(tmp_path, "bad", 300, 300)

    class Boom(FakeModel):
        def run(self, image, mask, cancel=None):
            raise RuntimeError("boom")

    rows = acc.run_all(Boom(), [c], tmp_path / "out", size=256, log=lambda s: None)
    assert rows[0]["error"].startswith("InferenceError")          # הכשל מדווח בשורה, ולא מפיל את שאר המקרים
    assert "כשל" in (tmp_path / "out" / "ACCEPT.md").read_text(encoding="utf-8")


def test_defaults_are_used(tmp_path):
    c = _case(tmp_path, "big", 1200, 1000)
    rows = acc.run_all(FakeModel(), [c], tmp_path / "o", size=384, log=lambda s: None)
    assert rows[0]["kinds"] in {"coarse", "downscale", "native", "tile"}
    assert PipelineConfig().coarse_belt_px == 0


def test_make_case_accepts_smaller_site_result_with_same_aspect(tmp_path):
    import subprocess
    import sys

    import cv2

    from fixpic.casefile import load_case
    from fixpic.imgio import save_image

    base = rnd(400, 300, seed=3)
    edited_full = base.copy()
    edited_full[100:300, 80:220] = 30                      # אזור "שהוסר"
    small = cv2.resize(edited_full, (150, 200), interpolation=cv2.INTER_AREA)
    save_image(tmp_path / "o.png", base)
    save_image(tmp_path / "site.png", small)
    tool = Path(__file__).resolve().parents[1] / "tools" / "m3" / "make_case.py"
    r = subprocess.run([sys.executable, str(tool), str(tmp_path / "o.png"), str(tmp_path / "site.png"), "-o", str(tmp_path / "c"),
                        "--reference", str(tmp_path / "site.png")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    case = load_case(tmp_path / "c")
    assert case["mask"].shape == (400, 300) and case["reference"].shape[:2] == (200, 150)
    assert (case["mask"][150:250, 100:200] > 0).all()
    bad = rnd(300, 300, seed=4)
    save_image(tmp_path / "b.png", bad)
    r2 = subprocess.run([sys.executable, str(tool), str(tmp_path / "o.png"), str(tmp_path / "b.png"), "-o", str(tmp_path / "c2")],
                        capture_output=True, text=True, encoding="utf-8")
    assert r2.returncode != 0 and "יחס" in (r2.stderr + r2.stdout)
