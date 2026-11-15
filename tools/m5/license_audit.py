"""M5 – סקירת רישיונות: סגור התלויות (runtime) של FIXPIC מתוך הסביבה המותקנת, עם רישיון, קטגוריה וסימון חומר נלווה.

שימוש (בסביבה שהותקנו בה התלויות, רצוי venv נקי):
  python tools/m5/license_audit.py                       # טבלה ל-stdout
  python tools/m5/license_audit.py --md out.md --json out.json
  python tools/m5/license_audit.py --check               # קוד יציאה 1 אם יש רישיון לא מסווג או copyleft חזק
  python tools/m5/license_audit.py --roots numpy pillow  # במקום התלויות מ-pyproject.toml

מה זה עושה: קורא את `dependencies` מ-`pyproject.toml`, עובר על `Requires-Dist` (בלי extras; סמני סביבה מחושבים עם `packaging` אם
מותקן, אחרת התלות נכללת ומסומנת), ולכל חבילה מוציא את הרישיון מ-`License-Expression`/`License`/Classifiers, מסווג אותו, ובודק את
קובצי הרישיון שבחבילה (dist-info) לאזכורי GPL/LGPL של רכיבים שנארזו בתוך הגלגל (למשל FFmpeg ב-OpenCV).

מגבלות (חשוב): (1) המטא-דאטה של חבילה אינו מכסה בהכרח כל רכיב שנארז בבינארים שלה; האזכורים בקובצי הרישיון הם רמז, לא סקירה
משפטית. (2) הסגור מחושב לפי הסביבה שבה הכלי רץ (מערכת הפעלה וגרסת Python); ב-Windows/3.14 ייתכנו תלויות נוספות.
(3) הסיווג הוא כלל אצבע; זה אינו ייעוץ משפטי."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

PERMISSIVE, WEAK, STRONG, UNKNOWN = "permissive", "weak-copyleft", "strong-copyleft", "unknown"
RANK = {PERMISSIVE: 0, WEAK: 1, STRONG: 2, UNKNOWN: 3}

# (תבנית, קטגוריה): הראשונה שמתאימה. LGPL לפני GPL כי "LGPL" מכיל "GPL".
_RULES = [
    (re.compile(r"\bAGPL", re.I), STRONG),
    (re.compile(r"\bLGPL|lesser general public", re.I), WEAK),
    (re.compile(r"\bGPL|general public licen[sc]e", re.I), STRONG),
    (re.compile(r"\bMPL|mozilla public", re.I), WEAK),
    (re.compile(r"\b(EPL|CDDL)\b|eclipse public", re.I), WEAK),
    (re.compile(r"\b(MIT|MIT-CMU|BSD(-\d-Clause)?|ISC|Apache([- ]2(\.0)?)?|Zlib|0BSD|CC0(-1\.0)?|HPND|PSF|Python|Unlicense|BlueOak|"
                r"MIT[- ]License|BSD[- ]License|3-Clause BSD|2-Clause BSD|Apache[- ]Software[- ]License|Apache[- ]License)\b", re.I), PERMISSIVE),
]


@dataclass
class Pkg:
    name: str
    version: str
    license: str                      # הטקסט המקורי (מקוצר)
    category: str
    note: str = ""                    # למשל "רישיון כפול: לבחור LGPL"
    license_files: list[str] = field(default_factory=list)
    mentions_lgpl: bool = False
    mentions_gpl: bool = False
    required_by: list[str] = field(default_factory=list)


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def classify_one(text: str) -> str:
    for rx, cat in _RULES:
        if rx.search(text):
            return cat
    return UNKNOWN


def classify(expr: str) -> tuple[str, str]:
    """(קטגוריה, הערה) לביטוי רישיון. `A OR B` = בחירה (הקטגוריה הנמוכה ביותר); `A AND B` = כולם חלים (הגבוהה ביותר)."""
    expr = (expr or "").strip()
    if not expr:
        return UNKNOWN, "אין מידע על רישיון במטא-דאטה"
    parts = [p.strip(" ()") for p in re.split(r"\s+OR\s+", expr)]
    if len(parts) > 1:
        cats = [classify(p)[0] for p in parts]
        best = min(cats, key=RANK.__getitem__)
        return best, f"רישיון כפול/מרובה: לבחור את האפשרות הפחות מגבילה ({parts[cats.index(best)]})"
    parts = [p.strip(" ()") for p in re.split(r"\s+AND\s+", expr)]
    if len(parts) > 1:
        cats = [classify_one(p) for p in parts]
        return max(cats, key=RANK.__getitem__), "כל הרישיונות חלים (AND)"
    return classify_one(expr), ""


def license_text(md) -> str:
    """הרישיון מהמטא-דאטה: License-Expression, אחרת License (אם קצר), אחרת Classifiers."""
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()
    lic = (md.get("License") or "").strip()
    if lic and len(lic) <= 120 and "\n" not in lic:
        return lic
    cl = [c.split("::")[-1].strip() for c in (md.get_all("Classifier") or []) if c.startswith("License")]
    cl = [c for c in cl if c.lower() != "osi approved"]
    return " OR ".join(cl) if len(cl) == 1 else " AND ".join(cl)


def root_requirements(pyproject: Path) -> list[str]:
    import tomllib
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return list(data.get("project", {}).get("dependencies", []))


def req_name(req: str) -> str:
    return canon(re.split(r"[\s;<>=!~\[(]", req.strip(), maxsplit=1)[0])


def marker_applies(req: str) -> tuple[bool, bool]:
    """(האם התלות חלה, האם הסמן נבדק בפועל). extras נדחים. בלי `packaging`: נכללת ולא נבדקה."""
    if ";" not in req:
        return True, True
    marker = req.split(";", 1)[1].strip()
    if re.search(r"\bextra\b", marker):
        return False, True
    try:
        from packaging.markers import Marker
        return Marker(marker).evaluate({"extra": ""}), True
    except Exception:
        return True, False


def _files_scan(dist) -> tuple[list[str], bool, bool]:
    files, lgpl, gpl = [], False, False
    for f in dist.files or []:
        low = str(f).lower()
        base = low.rsplit("/", 1)[-1]
        if not re.search(r"(licen[sc]e|copying|notice|third[-_]?party)", base):
            continue
        files.append(str(f))
        try:
            txt = Path(dist.locate_file(f)).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"\bLGPL|lesser general public", txt, re.I):
            lgpl = True
        if re.search(r"(?<!L)GPL|(?<!lesser )general public licen[sc]e", txt, re.I):
            gpl = True
    return files, lgpl, gpl


def collect(roots: list[str], dists: dict | None = None) -> tuple[list[Pkg], list[str]]:
    """סגור התלויות של `roots` (שמות או דרישות) לפי `dists` (שם מנורמל -> Distribution; ברירת מחדל: הסביבה). מחזיר (חבילות, אזהרות)."""
    if dists is None:
        dists = {}
        for d in metadata.distributions():
            n = d.metadata["Name"]
            if n:
                dists.setdefault(canon(n), d)
    warnings: list[str] = []
    out: dict[str, Pkg] = {}
    queue = [(req_name(r), "(שורש)") for r in roots]
    while queue:
        name, parent = queue.pop(0)
        if name in out:
            if parent not in out[name].required_by:
                out[name].required_by.append(parent)
            continue
        d = dists.get(name)
        if d is None:
            warnings.append(f"{name}: לא מותקנת בסביבה (נדרשת ע\"י {parent}); לא נסרקה")
            continue
        md = d.metadata
        text = license_text(md)
        cat, note = classify(text)
        files, lgpl, gpl = _files_scan(d)
        out[name] = Pkg(md["Name"], d.version, text[:120], cat, note, files, lgpl, gpl, [parent])
        for req in d.requires or []:
            ok, checked = marker_applies(req)
            if not ok:
                continue
            if not checked:
                warnings.append(f"{name} -> {req_name(req)}: סמן סביבה לא נבדק (חסרה `packaging`); נכלל")
            queue.append((req_name(req), md["Name"]))
    return sorted(out.values(), key=lambda p: (-RANK[p.category], p.name.lower())), warnings


CAT_HE = {PERMISSIVE: "מתירני", WEAK: "Copyleft חלש", STRONG: "Copyleft חזק", UNKNOWN: "לא ידוע"}


def to_markdown(pkgs: list[Pkg], warnings: list[str]) -> str:
    rows = ["| חבילה | גרסה | רישיון (מטא-דאטה) | קטגוריה | חומר נלווה | נדרשת ע\"י |", "|---|---|---|---|---|---|"]
    for p in pkgs:
        extra = []
        if p.mentions_lgpl:
            extra.append("אזכור LGPL בקובצי הרישיון")
        if p.mentions_gpl:
            extra.append("אזכור GPL בקובצי הרישיון")
        extra.append(f"{len(p.license_files)} קובצי רישיון" if p.license_files else "אין קובץ רישיון")
        lic = p.license + (f" ({p.note})" if p.note else "")
        rows.append(f"| {p.name} | {p.version} | {lic} | {CAT_HE[p.category]} | {'; '.join(extra)} | {', '.join(p.required_by)} |")
    out = "\n".join(rows)
    if warnings:
        out += "\n\nאזהרות:\n" + "\n".join(f"- {w}" for w in warnings)
    return out


def problems(pkgs: list[Pkg]) -> list[str]:
    bad = []
    for p in pkgs:
        if p.category in (STRONG, UNKNOWN):
            bad.append(f"{p.name} {p.version}: {CAT_HE[p.category]} ({p.license or 'ללא מידע'})")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[2] / "pyproject.toml")
    ap.add_argument("--roots", nargs="+", help="שמות חבילות שורש (ברירת מחדל: dependencies מ-pyproject.toml)")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--check", action="store_true", help="קוד יציאה 1 אם יש חבילה בקטגוריה Copyleft חזק או לא ידוע")
    a = ap.parse_args(argv)
    roots = a.roots or root_requirements(a.pyproject)
    pkgs, warnings = collect(roots)
    md = to_markdown(pkgs, warnings)
    print(f"Python {sys.version.split()[0]}, {sys.platform}, {len(pkgs)} חבילות")
    print(md)
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps({"python": sys.version.split()[0], "platform": sys.platform, "roots": roots,
                                      "packages": [asdict(p) for p in pkgs], "warnings": warnings}, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = problems(pkgs)
    if a.check and bad:
        print("\nבעיות:\n" + "\n".join(f"- {b}" for b in bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
