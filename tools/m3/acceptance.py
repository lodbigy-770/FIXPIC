"""M3.7.11: ערכת קבלה. מריצה את **ברירות המחדל של המוצר** (`PipelineConfig()`, בלי רשת תצורות) על מקרים, ולכל מקרה
שומרת תוצאה מלאה, גיליון צד-לצד (מקור עם המסכה באדום | ייחוס אם יש | התוצאה שלנו | התוצאה שלנו מוקטנת לגודל הייחוס)
והמדדים. בסוף נכתב `ACCEPT.md` עם טבלה ועמודת \"ממצא בעין\" למילוי.

הרצה (אצל המשתמש, על Big LaMa):
  python tools/m3/acceptance.py cases/boy_large cases/paris_chair cases/table_camera --out m3_accept
  python tools/m3/acceptance.py cases/NEW1 cases/NEW2 cases/NEW3 --out m3_accept        # הסט הנוסף (3-5 מסכות גדולות)
  python tools/m3/acceptance.py cases/boy_large --plan-only                              # רק תכנון: בלי מודל
  python tools/m3/acceptance.py cases/boy_large --size 512 --out m3_accept_512           # bucket אחר; ברירת המחדל: 384

השוואה מול Cleanup.pictures: לכל מקרה `reference.png` = התוצאה של האתר (נוצר עם `make_case.py --reference`, או מהממשק:
קובץ -> \"שמירת מקרה בדיקה…\"). כשהייחוס קטן מהמקור (למשל 682x1024), השוואה הוגנת היא בגודל אחד: התא הרביעי והעמודות
`ref_*` הם התוצאה שלנו מוקטנת לגודל הייחוס. המדדים אינם מחליפים עין; ההכרעה בגיליון.
את כל התיקייה `--out` יש להחזיר (תוצאה מלאה כ-JPG איכות 95, כדי לא להעלות PNG של 24MP)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import quality_bench as qb  # noqa: E402

from fixpic.casefile import load_case  # noqa: E402
from fixpic.errors import FixPicError  # noqa: E402
from fixpic.pipeline import Pipeline, PipelineConfig  # noqa: E402


def _overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    o = img.copy()
    sel = mask > 0
    o[sel] = (0.5 * o[sel] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return o


def _save_jpg(path: Path, rgb: np.ndarray, q: int = 95) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError(f"קידוד JPG נכשל: {path}")
    path.write_bytes(buf.tobytes())


def run_case(model, case_dir: str | Path, out_dir: str | Path, *, cfg: PipelineConfig | None = None,
             size: int | None = None, log=print) -> dict:
    case_dir = Path(case_dir)
    case = load_case(case_dir)
    img, mask, ref = case["rgb"], case["mask"], case["reference"]
    brush = (case["meta"].get("params") or {}).get("brush_px")
    name = case_dir.name
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape
    pipe = Pipeline(model, cfg or PipelineConfig())
    rep = pipe.plan(mask, size, brush)
    row: dict = dict(case=name, size=[w, h], mask_pct=round(100 * float((mask > 0).mean()), 1),
                     kinds=",".join(sorted({g.kind for g in rep.groups})), planned_runs=rep.runs,
                     scale=round(rep.min_scale, 3), reference_size=list(ref.shape[1::-1]) if ref is not None else None)
    t0 = time.perf_counter()
    try:
        res = pipe.run(img, mask, size, brush_px=brush)
    except FixPicError as e:
        row["error"] = f"{type(e).__name__}: {e}"
        log(f"{name}: כשל {row['error']}")
        return row
    row["seconds"] = round(time.perf_counter() - t0, 2)
    row["model_calls"] = res.model_calls
    row["peak_rss_mb"] = None if (v := qb.peak_rss_mb()) is None else round(v)
    eff = res.effective_mask
    row.update(qb.safe_measure(res.image, eff) or {})
    cy, cx = qb.crop_box(mask)
    cells = [("original+mask", _overlay(img, mask)[cy, cx])]
    ref_hw = ref.shape[:2] if ref is not None and ref.shape[:2] != (h, w) else None
    if ref is not None:
        rimg, rmask = (ref, qb.to_ref_size(ref, mask, ref_hw)[1]) if ref_hw else (ref, mask > 0)
        row.update({f"reference_{k}": v for k, v in (qb.safe_measure(rimg, rmask) or {}).items()})
        big = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC) if ref_hw else ref
        cells.append(("reference (Cleanup.pictures)", big[cy, cx]))
    cells.append(("ours (defaults)", res.image[cy, cx]))
    if ref_hw:
        oimg, omask = qb.to_ref_size(res.image, eff, ref_hw)
        row.update({f"ref_{k}": v for k, v in (qb.safe_measure(oimg, omask) or {}).items()})
        ry, rx = qb.crop_box(qb.to_ref_size(ref, mask, ref_hw)[1])
        qb.write_png(out / "ours_refsize.png", oimg)
        cells.append((f"ours @ {ref_hw[1]}x{ref_hw[0]}", oimg[ry, rx]))
    _save_jpg(out / "ours.jpg", res.image)
    qb.write_png(out / "ours_crop.png", res.image[cy, cx])
    qb.write_png_sheet(out / "sheet.jpg", cells, cols=len(cells), cell_w=420)
    (out / "result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{name}: {row['seconds']}s, {res.model_calls} הרצות, {row['kinds']}, texture={row.get('texture_ratio', float('nan')):.2f} "
        f"stripe={row.get('stripe_score', float('nan')):.1f} cdiff={row.get('boundary_color_diff', float('nan')):.1f}")
    return row


def _f(v, nd=2):
    return "-" if v is None or isinstance(v, str) else f"{v:.{nd}f}"


def accept_md(rows: list[dict], cfg: PipelineConfig, size: int | None) -> str:
    lines = [
        "# קבלה M3.7.11 (ברירות המחדל של המוצר)", "",
        f"תצורה: `PipelineConfig()` עם coarse_fine={cfg.coarse_fine}, coarse_size={cfg.coarse_size}, coarse_belt_px={cfg.coarse_belt_px}, "
        f"min_scale={cfg.min_scale}, large_strategy={cfg.large_strategy}; bucket {size or cfg.default_size}.", "",
        "| מקרה | גודל | מסכה % | מסלול | הרצות | זמן s | זיכרון MB | texture (ייחוס) | stripe | cdiff | ממצא בעין (למילוי) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['case']} | {r['size'][0]}x{r['size'][1]} | {r['mask_pct']} | {r['kinds']} | - | - | - | - | - | - | כשל: {r['error']} |")
            continue
        ref_tex = r.get("reference_texture_ratio")
        tex = f"{_f(r.get('texture_ratio'))} ({_f(ref_tex)})" if ref_tex is not None else _f(r.get("texture_ratio"))
        lines.append(f"| {r['case']} | {r['size'][0]}x{r['size'][1]} | {r['mask_pct']} | {r['kinds']} | {r['model_calls']} | {r['seconds']} | "
                     f"{r.get('peak_rss_mb') or '-'} | {tex} | {_f(r.get('stripe_score'), 1)} | {_f(r.get('boundary_color_diff'), 1)} |  |")
    lines += ["", "**מילוי:** לכל מקרה, מול `sheet.jpg` (מקור | ייחוס | שלנו | שלנו בגודל הייחוס): נקי משרידים? מרוח? תפרים? "
              "טוב/דומה/גרוע יותר מ-Cleanup.pictures בגודל אחד? זמן מקובל?", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--out", default="m3_accept")
    ap.add_argument("--size", type=int, default=None, help="bucket (ברירת מחדל: default_size)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--model", default="big-lama")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    cfg = PipelineConfig()
    if a.plan_only:
        from fixpic.models.base import ModelInfo

        class _Stub:
            info = ModelInfo(name="stub", multiple=8, min_size=64, max_size=4096)

        for c in a.cases:
            case = load_case(c)
            brush = (case["meta"].get("params") or {}).get("brush_px")
            rep = Pipeline(_Stub(), cfg).plan(case["mask"], a.size, brush)
            print(f"{Path(c).name}: {case['mask'].shape[1]}x{case['mask'].shape[0]}, "
                  f"{[(g.kind, round(g.scale, 3), g.windows) for g in rep.groups]}, {rep.runs} הרצות")
        return
    from fixpic.cli import DEFAULT_MODELS
    from fixpic.models import ModelManager

    model = ModelManager(a.models_dir or DEFAULT_MODELS).load(a.model, threads=a.threads)
    run_all(model, a.cases, a.out, cfg=cfg, size=a.size)


def run_all(model, cases, out_dir, *, cfg: PipelineConfig | None = None, size: int | None = None, log=print) -> list[dict]:
    cfg = cfg or PipelineConfig()
    rows = [run_case(model, c, out_dir, cfg=cfg, size=size, log=log) for c in cases]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "ACCEPT.md").write_text(accept_md(rows, cfg, size), encoding="utf-8")
    (Path(out_dir) / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
