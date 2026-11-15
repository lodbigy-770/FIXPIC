"""M3.7.4 (ניתוח וכיול): קורא תיקיית פלט של `quality_bench.py` ומחשב מחדש את המדדים בתנאים שווים, כך שאפשר לדרג
תצורות ולהשוות לייחוס.

הרצה: `python tools/m3/analyze_bench.py m3_bench [--cases cases] [--out m3_bench/analysis.txt]`

למה מחשבים מחדש: (1) התוצאה שלנו נמדדה ב-`quality_bench` על המסכה האפקטיבית (מורחבת), והייחוס על מסכת המקרה; כאן הכול על
מסכת המקרה. (2) מוסיפים עמודות יחסיות (מול הייחוס ומול הטוב ביותר באותו מקרה), כי הערכים המוחלטים תלויי-תמונה (ראו להלן).
התוצאה המשוחזרת: המקור עם החיתוך שנשמר (`crops/<id>.png`) מודבק במקומו.

כללי הפרשנות שכוילו על שלושת המקרים (ראו DEVELOPMENT_LOG, M3.7.4; מעט נתונים, לא מסקנה סופית):
* `texture_ratio` (tex): מתחת ל-0.5 = מרוח (נראה כצל); מעל 0.85 = תקין. בין לבין: להסתכל. כשהייחוס עצמו נמוך (שולחן עם עץ:
  0.45), מעריכים ביחס אליו.
* `seam_ratio` מעל 1.3 = תפר קשה בקו התפר **הנכון** (הקווים מחושבים מהתכנון של המקרה; בריצת ממשק עם מסכה אחרת הם יזוזו).
* `stripe_score` אינו מדד מוחלט: על תמונה עם מבנים חזקים (רגלי שולחן) יוצא 100+ גם בלי תפר (והייחוס 135), ועל תפר אמיתי
  בריצת ממשק יצא 70. משתמשים בו יחסית: פי 3 ומעלה מהערך הנמוך באותו מקרה (או מהייחוס) = חשד לתפר.
* `boundary_color_diff` כמעט אינו מבדיל בין תצורות (הפרשים מתחת ל-1 ברמה), ומשקף בעיקר את הסצנה; יחס לייחוס בלבד."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fixpic.casefile import load_case  # noqa: E402
from fixpic.errors import FixPicError  # noqa: E402
from fixpic.quality import measure  # noqa: E402
from quality_bench import crop_box, to_ref_size  # noqa: E402

TEX_SMEAR, TEX_OK, SEAM_HARD, STRIPE_REL = 0.5, 0.85, 1.3, 3.0


def _m(img, mask):
    try:
        return measure(img, mask).as_dict()
    except FixPicError:
        return None


def verdict(row: dict, min_stripe: float, ref: dict | None) -> str:
    notes = []
    tex = row["texture_ratio"]
    ref_tex = ref["texture_ratio"] if ref else None
    if tex < TEX_SMEAR and not (ref_tex is not None and ref_tex < TEX_SMEAR and tex >= 0.9 * ref_tex):
        notes.append("מרוח")
    elif tex < TEX_OK and not (ref_tex is not None and tex >= 0.9 * ref_tex):
        notes.append("בינוני")
    if row.get("seam_ratio") is not None and row["seam_ratio"] >= SEAM_HARD:
        notes.append("תפר (seam)")
    elif min_stripe > 0 and row["stripe_score"] >= STRIPE_REL * min_stripe and row["stripe_score"] >= 12:
        notes.append("חשד לתפר (stripe)")
    return ", ".join(notes) or "תקין"


def analyze_case(bench: Path, cases: Path) -> list[str]:
    res = json.loads((bench / "results.json").read_text(encoding="utf-8"))
    name = res["case"]
    case = load_case(cases / name)
    img, mask = case["rgb"], case["mask"]
    cy, cx = crop_box(mask)
    ref = case["reference"]
    ref_hw = ref.shape[:2] if ref is not None and ref.shape[:2] != mask.shape else None
    ref_m = None
    if ref is not None:
        ref_m = _m(ref, to_ref_size(ref, mask, ref_hw)[1] if ref_hw else mask)
    rows = []
    for r in res["rows"]:
        if r.get("reference") or r.get("error"):
            continue
        p = bench / "crops" / f"{r['id']}.png"
        crop = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        full = img.copy()
        full[cy, cx] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        m = _m(full, mask)
        if m is None:
            continue
        m.update(id=r["id"], seconds=r.get("seconds"), runs=r["planned_runs"], seam_ratio=r.get("seam_ratio"))
        if ref_hw:
            rimg, rmask = to_ref_size(full, mask, ref_hw)
            m["tex_at_ref"] = (_m(rimg, rmask) or {}).get("texture_ratio")
        rows.append(m)
    min_stripe = min((r["stripe_score"] for r in rows), default=0.0)
    L = [f"== {name}: {mask.shape[1]}x{mask.shape[0]}" + (f", ייחוס {ref.shape[1]}x{ref.shape[0]}" if ref is not None else ", ללא ייחוס")]
    hdr = f"{'id':28} {'runs':>4} {'sec':>6} {'tex':>5} {'tex@ref':>7} {'strp':>6} {'strp/min':>8} {'seam':>5} {'cdiff':>6} {'cd/ref':>6}  פסק"
    L.append(hdr)
    if ref_m:
        L.append(f"{'reference':28} {'':>4} {'':>6} {ref_m['texture_ratio']:>5.2f} {'':>7} {ref_m['stripe_score']:>6.1f} {'':>8} {'':>5} "
                 f"{ref_m['boundary_color_diff']:>6.1f} {'1.00':>6}")
    for m in rows:
        cd_rel = f"{m['boundary_color_diff'] / ref_m['boundary_color_diff']:.2f}" if ref_m and not ref_hw else "-"
        tex_ref = f"{m['tex_at_ref']:.2f}" if m.get("tex_at_ref") is not None else "-"
        seam = f"{m['seam_ratio']:.2f}" if m["seam_ratio"] is not None else "-"
        sec = f"{m['seconds']:.1f}" if m["seconds"] is not None else "-"
        L.append(f"{m['id']:28} {m['runs']:>4} {sec:>6} {m['texture_ratio']:>5.2f} {tex_ref:>7} {m['stripe_score']:>6.1f} "
                 f"{m['stripe_score'] / min_stripe if min_stripe else 0:>8.1f} {seam:>5} {m['boundary_color_diff']:>6.1f} {cd_rel:>6}  "
                 f"{verdict(m, min_stripe, ref_m if not ref_hw else None)}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", type=Path)
    ap.add_argument("--cases", type=Path, default=Path("cases"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out: list[str] = []
    for d in sorted(p for p in a.bench.iterdir() if (p / "results.json").is_file()):
        out += analyze_case(d, a.cases) + [""]
    text = "\n".join(out)
    print(text)
    (a.out or a.bench / "analysis.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
