"""M5: בדיקת Latency מול התקציב (`tools/m5/latency_budget.py`), על מודלים סינתטיים ותמונות קטנות."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latency_budget", REPO / "tools" / "m5" / "latency_budget.py")
lb = importlib.util.module_from_spec(spec)
sys.modules["latency_budget"] = lb
spec.loader.exec_module(lb)


class SlowModel(lb.SyntheticModel):
    """מדמה מודל אמיתי איטי (השהיה קבועה לכל הרצה)."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay
        self.warmed: list = []

    def run(self, image, mask, cancel=None):
        time.sleep(self.delay)
        return super().run(image, mask, cancel)

    def warmup(self, sizes, cancel=None):
        self.warmed = list(sizes)


class FailingModel(lb.SyntheticModel):
    def run(self, image, mask, cancel=None):
        from fixpic.errors import InferenceError
        raise InferenceError("כשל מדומה")


def _run(model, label="fake", sizes=(256,), masks=("stroke",), reps=2, warm=True):
    return lb.run_benchmark(model, label, 3, 1.0, list(sizes), list(masks), reps, warm=warm)


def test_budgets_match_m15_decision():
    assert lb.BUDGETS[256][0] == 0.8 and lb.BUDGETS[384][0] == 2.0 and lb.BUDGETS[512][0] == 4.0
    assert lb.BUDGETS[768][0] is None


def test_make_image_and_mask_shapes_and_determinism():
    a, b = lb.make_image(60, 40), lb.make_image(60, 40)
    assert a.shape == (40, 60, 3) and a.dtype.name == "uint8" and (a == b).all()
    m = lb.make_mask(60, 40, 10)
    assert m.shape == (40, 60) and int((m > 0).sum()) == 100
    assert int((lb.make_mask(60, 40, 500) > 0).sum()) == 40 * 40      # נחתך לתמונה


def test_verdict_for():
    assert lb.verdict_for(0.5, 0.8, True) is True
    assert lb.verdict_for(0.9, 0.8, True) is False
    assert lb.verdict_for(0.8, 0.8, True) is True                      # גבול כולל
    assert lb.verdict_for(0.1, 0.8, False) is None                     # סינתטי
    assert lb.verdict_for(5.0, None, True) is None                     # בלי תקציב
    assert lb.verdict_for(None, 0.8, True) is None


def test_synthetic_model_has_no_verdict_and_exit_zero():
    r = _run(lb.SyntheticModel(), label="synthetic", sizes=(256, 384), masks=("stroke", "region"))
    assert len(r["rows"]) == 4
    assert all(row["ok"] and row["verdict"] is None for row in r["rows"])
    assert lb.exit_code(r) == 0
    assert "אין פסק דין" in lb.to_markdown(r)


def test_slow_model_over_budget_fails_with_exit_one():
    r = _run(SlowModel(0.9))                       # 256: תקציב 0.8s
    row = r["rows"][0]
    assert row["ok"] and row["median_s"] >= 0.9 and row["verdict"] is False
    assert lb.exit_code(r) == 1


def test_fast_model_within_budget_passes():
    r = _run(SlowModel(0.05))
    assert r["rows"][0]["verdict"] is True and lb.exit_code(r) == 0


def test_bucket_without_budget_never_fails():
    r = _run(SlowModel(0.05), sizes=(768,))
    row = r["rows"][0]
    assert row["budget_s"] is None and row["verdict"] is None and lb.exit_code(r) == 0


def test_failure_is_recorded_and_other_scenarios_continue():
    r = _run(FailingModel(), label="fake", sizes=(256, 384))
    assert len(r["rows"]) == 2
    assert all(not row["ok"] and row["verdict"] is False and "InferenceError" in row["error"] for row in r["rows"])
    assert lb.exit_code(r) == 1
    assert "חרג/נכשל" in lb.to_markdown(r)


def test_warmup_called_with_sizes_and_skippable():
    m = SlowModel(0)
    _run(m, sizes=(384, 256))
    assert m.warmed == [256, 384]
    m2 = SlowModel(0)
    r = _run(m2, warm=False)
    assert m2.warmed == [] and r["info"]["warmup_s"] is None


def test_reps_counted_and_warmup_run_discarded():
    calls = []

    class Counting(lb.SyntheticModel):
        def run(self, image, mask, cancel=None):
            calls.append(1)
            return super().run(image, mask, cancel)

    r = _run(Counting(), reps=3)
    assert r["rows"][0]["n"] == 3 and len(calls) == 4                  # 1 חימום + 3 מדודות


def test_info_has_hardware_fields():
    i = _run(lb.SyntheticModel(), label="synthetic")["info"]
    for k in ("threads", "cpu_count", "platform", "python", "image_mp", "reps", "backend"):
        assert k in i
    assert i["threads"] == 3 and i["backend"] == "synthetic"


def test_cli_synthetic_writes_json_and_md(tmp_path):
    j, md = tmp_path / "o.json", tmp_path / "o.md"
    rc = lb.main(["--mp", "0.5", "--reps", "1", "--sizes", "256", "--masks", "stroke", "--json", str(j), "--md", str(md)])
    assert rc == 0
    data = json.loads(j.read_text(encoding="utf-8"))
    assert data["info"]["model"] == "synthetic" and len(data["rows"]) == 1
    assert "Latency" in md.read_text(encoding="utf-8")


def test_cli_rejects_unknown_mask():
    with pytest.raises(SystemExit):
        lb.main(["--masks", "nope"])
