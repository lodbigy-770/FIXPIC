"""M5: מחולל THIRD_PARTY_NOTICES (`tools/m5/third_party_notices.py`) על מטא-דאטה מדומה."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("third_party_notices", REPO / "tools" / "m5" / "third_party_notices.py")
tpn = importlib.util.module_from_spec(spec)
sys.modules["third_party_notices"] = tpn
spec.loader.exec_module(tpn)
la = tpn.la


class _D:
    def __init__(self, root, files):
        self._root, self.files = root, files

    def locate_file(self, f):
        return self._root / f


def _pkg(name, lic="MIT"):
    return la.Pkg(name, "1.0", lic, la.PERMISSIVE)


def test_package_texts_picks_only_license_files(tmp_path):
    for n, t in {"LICENSE.txt": "mit text", "NOTICE": "notice", "mod.py": "code", "license_helper.py": "x", "third_party.h": "h"}.items():
        (tmp_path / n).write_text(t, encoding="utf-8")
    got = tpn.package_texts(_D(tmp_path, ["LICENSE.txt", "NOTICE", "mod.py", "license_helper.py", "third_party.h", "missing/LICENSE"]))
    assert [p for p, _ in got] == ["LICENSE.txt", "NOTICE"]


def test_render_dedupes_identical_text_and_warns_on_missing():
    pkgs = [_pkg("a"), _pkg("b"), _pkg("c")]
    texts = {"a": [("LICENSE", "same text")], "b": [("LICENSE", "same text\n")], "c": []}
    warnings: list[str] = []
    out = tpn.render(pkgs, lambda p: texts[p.name], [("Extra", "extra text")], warnings)
    assert out.count("same text") == 1 and "identical to a: LICENSE" in out
    assert "extra text" in out and "- a 1.0: MIT" in out
    assert warnings == ["c: אין קובץ רישיון בחבילה"]


def test_render_extra_identical_to_package_text_is_referenced():
    out = tpn.render([_pkg("a")], lambda p: [("LICENSE", "LGPL body")], [("LGPL", "LGPL body")], [])
    assert out.count("LGPL body") == 1 and "identical to a: LICENSE" in out


def test_required_license_files_exist_in_repo():
    extras = tpn.default_extras()
    titles = " ".join(t for t, _ in extras)
    assert "Apache-2.0" in titles and "Lesser General Public License v3.0" in titles and "General Public License v3.0" in titles
    assert all(len(txt) > 5000 for _, txt in extras)
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in extras[1][1].upper() and "Version 3" in extras[1][1]


def test_main_writes_file_and_excludes(tmp_path, monkeypatch):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[project]\nname="x"\nversion="0"\ndependencies=["numpy","packaging"]\n', encoding="utf-8")
    out = tmp_path / "sub" / "N.txt"
    assert tpn.main(["--pyproject", str(pp), "--out", str(out), "--exclude", "packaging"]) == 0
    txt = out.read_text(encoding="utf-8")
    assert txt.startswith("THIRD-PARTY NOTICES FOR FIXPIC") and "- numpy " in txt and "- packaging " not in txt
    assert tpn.DEFAULT_EXCLUDE == ("openvino-telemetry",)
