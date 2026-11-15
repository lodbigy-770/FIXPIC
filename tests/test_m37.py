"""M3.7.1: מקרה בדיקה (`fixpic.casefile`), שחזור מסכה מהפרש, ופעולת "שמירת מקרה בדיקה" בממשק. Qt offscreen."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from fixpic.casefile import derive_mask, load_case, read_mask, save_case
from fixpic.errors import InvalidInputError
from test_engine import rnd

REPO = Path(__file__).resolve().parents[1]


def test_save_and_load_case_roundtrip_in_hebrew_folder(tmp_path):
    rgb = rnd(120, 160, 1)
    mask = np.zeros((120, 160), np.uint8)
    mask[30:80, 40:100] = 255
    res = rnd(120, 160, 2)
    d = save_case(tmp_path / "תיקייה" / "מקרה א", mask=mask, original_rgb=rgb, result_rgb=res,
                  params={"brush_px": 20.5, "size": None}, note="בדיקה")
    c = load_case(d)
    assert np.array_equal(c["rgb"], rgb) and np.array_equal(c["mask"], mask) and np.array_equal(c["result"], res)
    assert c["reference"] is None and c["meta"]["params"]["brush_px"] == 20.5
    assert c["meta"]["mask_pixels"] == 50 * 60 and c["meta"]["note"] == "בדיקה"


def test_case_with_original_ref_and_validation(tmp_path):
    from fixpic.imgio import save_image
    rgb = rnd(60, 80, 3)
    save_image(tmp_path / "big.png", rgb)
    mask = np.zeros((60, 80), np.uint8)
    mask[10:20, 10:20] = 255
    d = save_case(tmp_path / "c", mask=mask, original_ref="../big.png")
    assert np.array_equal(load_case(d)["rgb"], rgb)
    with pytest.raises(InvalidInputError):
        save_case(tmp_path / "x", mask=mask)                                   # אין מקור
    with pytest.raises(InvalidInputError):
        save_case(tmp_path / "y", mask=mask, original_rgb=rnd(10, 10))         # מידות שונות
    meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
    meta["schema"] = 99
    (d / "case.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(InvalidInputError):
        load_case(d)


def test_derive_mask_recovers_edited_region_and_fills_holes():
    h, w = 300, 400
    a = rnd(h, w, 4)
    b = a.copy()
    true = np.zeros((h, w), bool)
    true[80:220, 100:300] = True
    b[true] = 128
    b[140:150, 190:200] = a[140:150, 190:200]            # חור פנימי שבו התוצאה זהה למקור
    m = derive_mask(a, b) > 0
    assert m[(80 + 5):(220 - 5), (100 + 5):(300 - 5)].all()                   # כולל החור
    inter, union = (m & true).sum(), (m | true).sum()
    assert inter / union > 0.9
    assert not m[:40].any()
    with pytest.raises(InvalidInputError):
        derive_mask(a, b[:100])


@pytest.mark.skipif(not (REPO / "cases").exists(), reason="cases/ לא קיים")
@pytest.mark.parametrize("name", ["paris_chair", "table_camera", "boy_large"])
def test_checked_in_cases_load_with_consistent_masks(name):
    d = REPO / "cases" / name
    meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
    mask = read_mask(d / "mask.png")
    assert mask.shape == (meta["height"], meta["width"]) and int((mask > 0).sum()) == meta["mask_pixels"]
    assert 0.05 < (mask > 0).mean() < 0.3


# ------------------------------------------------------------------ ממשק
pytest.importorskip("PySide6")
from test_m34 import (EWin, app, drag, wait_until, ew, T, _close_canvases)  # noqa: E402,F401


@pytest.mark.usefixtures("_close_canvases")
def test_ui_save_case_writes_last_submission(ew, tmp_path):
    assert ew.ready()
    s = ew.install(h=200, w=300)
    errs = ew.errors
    ew.w.save_case()
    assert errs and errs[-1][0] == "msg.no_case"                              # עוד לא הוגשה משיכה
    before = s.rgb.copy()
    drag(ew.canvas, [(100, 100), (140, 100)])
    assert ew.idle() and s.last_submission is not None
    sub = s.last_submission
    assert np.array_equal(sub.rgb, before) and sub.mask.any() and sub.brush_px is not None
    ew.w._pick_dir = lambda: str(tmp_path)
    ew.w.save_case()
    assert wait_until(lambda: not ew.w.busy and any(tmp_path.iterdir()))
    d = next(tmp_path.iterdir())
    c = load_case(d)
    assert np.array_equal(c["rgb"], before) and np.array_equal(c["mask"] > 0, sub.mask > 0)
    assert np.array_equal(c["result"], s.rgb) and c["meta"]["params"]["brush_px"] == sub.brush_px
