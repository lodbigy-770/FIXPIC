"""M5: סקירת רישיונות (`tools/m5/license_audit.py`): סיווג, סגור תלויות על מטא-דאטה מדומה, ופלט."""
from __future__ import annotations

import importlib.util
import json
import sys
from email.message import Message
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("license_audit", REPO / "tools" / "m5" / "license_audit.py")
la = importlib.util.module_from_spec(spec)
sys.modules["license_audit"] = la          # נדרש ל-dataclass עם annotations
spec.loader.exec_module(la)


class FakeDist:
    def __init__(self, name, version="1.0", license_expr=None, license_field=None, classifiers=(), requires=(), files=None, root=None):
        m = Message()
        m["Name"] = name
        if license_expr:
            m["License-Expression"] = license_expr
        if license_field:
            m["License"] = license_field
        for c in classifiers:
            m["Classifier"] = c
        self.metadata, self.version, self.requires = m, version, list(requires)
        self._root, self.files = root, files

    def locate_file(self, f):
        return self._root / f


@pytest.mark.parametrize("expr,cat", [
    ("MIT", la.PERMISSIVE), ("Apache-2.0", la.PERMISSIVE), ("BSD-3-Clause", la.PERMISSIVE), ("MIT-CMU", la.PERMISSIVE),
    ("LGPL-3.0-only", la.WEAK), ("MPL-2.0", la.WEAK), ("GPL-3.0-only", la.STRONG), ("AGPL-3.0-or-later", la.STRONG),
    ("", la.UNKNOWN), ("Proprietary EULA", la.UNKNOWN),
])
def test_classify_single(expr, cat):
    assert la.classify(expr)[0] == cat


def test_lgpl_is_not_mistaken_for_gpl():
    assert la.classify_one("LGPL-2.1-or-later") == la.WEAK


def test_classify_or_picks_least_restrictive_and_and_picks_most():
    cat, note = la.classify("LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only")
    assert cat == la.WEAK and "LGPL-3.0-only" in note
    assert la.classify("MIT OR GPL-3.0-only")[0] == la.PERMISSIVE
    assert la.classify("BSD-3-Clause AND GPL-3.0-or-later")[0] == la.STRONG
    assert la.classify("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0")[0] == la.PERMISSIVE


def test_license_text_sources():
    assert la.license_text(FakeDist("a", license_expr="MIT").metadata) == "MIT"
    assert la.license_text(FakeDist("a", license_field="Apache 2.0").metadata) == "Apache 2.0"
    long = FakeDist("a", license_field="x" * 500, classifiers=["License :: OSI Approved :: MIT License"])
    assert la.license_text(long.metadata) == "MIT License"
    assert la.license_text(FakeDist("a").metadata) == ""


def test_req_name_and_markers():
    assert la.req_name("PySide6_Essentials==6.12.0 ; extra == 'x'") == "pyside6-essentials"
    assert la.req_name("onnxruntime>=1.30") == "onnxruntime"
    assert la.marker_applies("foo>=1") == (True, True)
    assert la.marker_applies("foo; extra == 'dev'") == (False, True)
    assert la.marker_applies("foo; python_version < '3.0'")[0] is False
    assert la.marker_applies("foo; sys_platform == 'win32' and python_version >= '3.8'")[1] in (True, False)


def test_collect_closure_dedupes_and_records_parents(tmp_path):
    lic = tmp_path / "LICENSE.txt"
    lic.write_text("Linked with the GNU Lesser General Public License.", encoding="utf-8")
    dists = {
        "root": FakeDist("Root", license_expr="MIT", requires=["child>=1", "other", "ghost; extra == 'x'", "never; python_version < '3.0'"]),
        "child": FakeDist("Child", license_expr="GPL-3.0-only", requires=["other"], files=["LICENSE.txt"], root=tmp_path),
        "other": FakeDist("Other", license_field="Apache 2.0"),
    }
    pkgs, warnings = la.collect(["Root"], dists)
    names = [p.name for p in pkgs]
    assert sorted(names) == ["Child", "Other", "Root"] and names[0] == "Child"       # חמור ראשון
    other = next(p for p in pkgs if p.name == "Other")
    assert sorted(other.required_by) == ["Child", "Root"]
    child = next(p for p in pkgs if p.name == "Child")
    assert child.mentions_lgpl and child.license_files == ["LICENSE.txt"]
    assert warnings == []


def test_collect_warns_on_missing_distribution():
    pkgs, warnings = la.collect(["a"], {"a": FakeDist("A", license_expr="MIT", requires=["missing"])})
    assert [p.name for p in pkgs] == ["A"] and any("missing" in w for w in warnings)


def test_problems_and_markdown():
    pkgs, _ = la.collect(["a"], {"a": FakeDist("A", license_expr="GPL-3.0-only", requires=["b"]), "b": FakeDist("B")})
    bad = la.problems(pkgs)
    assert len(bad) == 2 and any("A 1.0" in b for b in bad)
    md = la.to_markdown(pkgs, ["אזהרה"])
    assert md.startswith("| חבילה |") and "Copyleft חזק" in md and "לא ידוע" in md and "- אזהרה" in md


def test_main_check_flag_and_outputs(tmp_path, monkeypatch, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname="x"\nversion="0"\ndependencies=["numpy"]\n', encoding="utf-8")
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    assert la.main(["--pyproject", str(pyproject), "--md", str(md), "--json", str(js)]) == 0
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["roots"] == ["numpy"] and any(p["name"].lower() == "numpy" for p in data["packages"])
    assert md.read_text(encoding="utf-8").startswith("| חבילה |")
    monkeypatch.setattr(la, "problems", lambda pkgs: ["x"])
    assert la.main(["--pyproject", str(pyproject), "--check"]) == 1
    assert "בעיות" in capsys.readouterr().err
