"""M5: בדיקת תיקיית האריזה (`tools/m5/check_dist.py`) על עצי קבצים מדומים, ובדיקת קיום ה-spec והחריגים."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_dist", REPO / "tools" / "m5" / "check_dist.py")
cd = importlib.util.module_from_spec(spec)
sys.modules["check_dist"] = cd
spec.loader.exec_module(cd)


def make_dist(root: Path, *, notices=20_000, model=2_000_000, extra=(), exe=True, qt=True) -> Path:
    def w(rel, size):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    if exe:
        w("FixPic", 1000)
    if notices is not None:
        w("_internal/THIRD_PARTY_NOTICES.txt", notices)
    if model is not None:
        w("_internal/models/big-lama/inpainter.onnx", model)
        w("_internal/models/big-lama/model.json", 300)
    if qt:
        w("_internal/PySide6/Qt/lib/libQt6Core.so.6", 500)
    for e in extra:
        w(e, 10)
    return root


def status(rep, name):
    return next(c["status"] for c in rep["checks"] if c["name"].startswith(name))


def test_good_dist_all_ok(tmp_path):
    rep = cd.inspect_dist(make_dist(tmp_path))
    assert [c["status"] for c in rep["checks"]] == ["ok"] * 6 and rep["files"] == 5


def test_missing_or_tiny_notices_fail(tmp_path):
    assert status(cd.inspect_dist(make_dist(tmp_path / "a", notices=None)), "THIRD") == "fail"
    assert status(cd.inspect_dist(make_dist(tmp_path / "b", notices=100)), "THIRD") == "fail"


def test_forbidden_components_detected(tmp_path):
    root = make_dist(tmp_path, extra=["_internal/openvino_telemetry/main.py", "_internal/PySide6/Qt/lib/libQt6WebEngineCore.so.6",
                                      "_internal/openvino/tools/ovc/main.py"])
    rep = cd.inspect_dist(root)
    c = next(c for c in rep["checks"] if c["name"].startswith("רכיבים"))
    assert c["status"] == "fail" and "3 קבצים" in c["detail"]


def test_lfs_pointer_model_fails_and_missing_model_fails(tmp_path):
    assert status(cd.inspect_dist(make_dist(tmp_path / "a", model=134)), "מודל") == "fail"
    assert status(cd.inspect_dist(make_dist(tmp_path / "b", model=None)), "מודל") == "fail"


def test_no_exe_or_no_qt_fails(tmp_path):
    assert status(cd.inspect_dist(make_dist(tmp_path / "a", exe=False)), "קישור") == "fail"
    assert status(cd.inspect_dist(make_dist(tmp_path / "b", qt=False)), "קישור") == "fail"


def test_main_exit_codes_and_json(tmp_path):
    good = make_dist(tmp_path / "g")
    out = tmp_path / "o.json"
    assert cd.main([str(good), "--json", str(out)]) == 0 and json.loads(out.read_text(encoding="utf-8"))["files"] == 5
    assert cd.main([str(make_dist(tmp_path / "b", model=1))]) == 1
    assert cd.main([str(tmp_path / "nope")]) == 2


def test_smoke_fails_when_process_exits_by_itself(tmp_path, monkeypatch):
    root = make_dist(tmp_path)
    (root / "FixPic").write_text("#!/bin/sh\nexit 3\n")
    (root / "FixPic").chmod(0o755)
    r = cd.smoke(root, 5)
    assert r["status"] == "fail" and "קוד 3" in r["detail"]


def test_smoke_ok_when_process_stays_alive(tmp_path):
    root = make_dist(tmp_path)
    (root / "FixPic").write_text("#!/bin/sh\nexec sleep 30\n")
    (root / "FixPic").chmod(0o755)
    assert cd.smoke(root, 1)["status"] == "ok"


def test_spec_matches_license_decisions():
    s = (REPO / "fixpic-app.spec").read_text(encoding="utf-8")
    for must in ("openvino_telemetry", "openvino.tools", "PySide6.QtWebEngineCore", "PySide6.QtPdf"):
        assert f"'{must}'" in s
    assert "upx=False" in s and "COLLECT(" in s and "exclude_binaries=True" in s and "THIRD_PARTY_NOTICES" in s
    assert (REPO / "packaging" / "fixpic_app.py").is_file()
    assert all(re.search(rf"'{k}'", s) for k in ("models", "licenses"))
