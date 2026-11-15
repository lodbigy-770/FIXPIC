"""M3.7.4: ערכת השוואה לאיכות הסרה באזורים גדולים. מריצה מקרה (תמונה + מסכה) על רשת תצורות של ה-Pipeline, ולכל תצורה
שומרת חיתוך של האזור, זמן, שיא זיכרון והמדדים מ-`fixpic.quality` (צל מטושטש, פסים, צבע בגבול, תפרי אריחים),
וגיליון השוואה אחד (תמונה אחת עם חיתוך לכל תצורה, ובתחילתו המקור והייחוס אם יש).

הרצה (אצל המשתמש, על Big LaMa):
  python tools/m3/quality_bench.py cases/boy_large --out m3_bench                    # הרשת הברירת-מחדל
  python tools/m3/quality_bench.py cases/boy_large --plan-only                       # רק תכנון: אילו תצורות ובכמה הרצות, בלי מודל
  python tools/m3/quality_bench.py cases/paris_chair cases/table_camera --out m3_bench --quick
  python tools/m3/quality_bench.py cases/boy_large --out m3_bench --heavy                # M3.7.5: חלון יחיד ב-768/1024/1536
  python tools/m3/quality_bench.py cases/boy_large --sizes 384,512 --strategies auto,tile --min-scales 0.5,0.25 \\
        --feathers 3,8 --overlaps 64,128 --blends 0,32 --coarses 0,1 --max-runs 60 --threads 3
  python tools/m3/quality_bench.py cases/boy_large --out m3_bench --sizes 384 --strategies auto --min-scales 0.5 --blends 32 \\
        --coarses 1 --coarse-sizes 768 --belt-px 0 --textures 0,1   # M3.7.8 על מסלול ברירת המחדל (חלון גס 768 בלי חגורה)
  #   # M3.7.8: שחזור טקסטורה (הוסיף "-tx" למזהה)

הרשת היא מכפלה של: `--sizes` x `--strategies` x `--min-scales` (רק ל-auto) x `--feathers` x `--overlaps` x `--blends` (רק כשיש אריחים; `tile_blend_px`, 0 = תפר קשה).
תצורות שהתכנון שלהן זהה (אותו סוג/קנה מידה/מספר הרצות לכל קבוצה) מורצות פעם אחת. תצורה שדורשת יותר מ-`--max-runs` הרצות
מודל מדולגת ומדווחת (ב-i5-6500T כל הרצה כ-1-2 שניות). שיא הזיכרון הוא של התהליך כולו עד כה (עולה בלבד), ולכן ערך גבוה
בתצורה מסוימת אומר שהיא העלתה את השיא; לבידוד מלא מריצים תצורה אחת בכל הרצה.

פלט ב-`--out/<שם המקרה>/`: `results.json`, `results.csv`, `summary.txt`, `sheet.jpg`, ו-`crops/<id>.png`. עם `--save-full` גם
התמונה המלאה לכל תצורה. את כל התיקייה יש להחזיר.

הערת מדדים: ערכי `texture_ratio` ו-`stripe_score` תלויי-תמונה, ולכן משווים בין תצורות על אותו מקרה ולא בין מקרים. כשיש
`reference.png` בגודל אחר (למשל 682x1024 מול 3456x5184), המדדים מחושבים גם על התוצאה מוקטנת לגודל הייחוס (עמודות `ref_*`),
כדי להשוות בגודל אחד מול הייחוס עצמו (שורת reference). המדדים אינם מחליפים עין: הגיליון הוא העיקר."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fixpic.casefile import load_case  # noqa: E402
from fixpic.errors import FixPicError  # noqa: E402
from fixpic.pipeline import Pipeline, PipelineConfig  # noqa: E402
from fixpic.quality import measure, seam_ratio  # noqa: E402

DEFAULT_GRID = dict(sizes=[384, 512], strategies=["downscale", "auto", "tile"], min_scales=[0.5, 0.25], feathers=[3],
                    overlaps=[64], blends=[0, 32])
# M3.7.5: חלון יחיד גדול (הקטנה בהרצה אחת). ~10-45 שניות להרצה לפי המדידות ב-latency_results.json (768: 10s, 1024: 18s, 4 threads)
HEAVY_GRID = dict(sizes=[768, 1024, 1536], strategies=["downscale"], min_scales=[0.5], feathers=[3], overlaps=[64],
                  blends=[32], coarses=[0])
QUICK_GRID = dict(sizes=[384], strategies=["auto", "tile"], min_scales=[0.5], feathers=[3], overlaps=[64], blends=[0, 32])


# ---------------------------------------------------------------- עזר

def peak_rss_mb() -> float | None:
    """שיא הזיכרון של התהליך עד כה (MB), או None אם אי אפשר למדוד. ב-Windows: PeakWorkingSetSize (psutil אם מותקן,
    אחרת ctypes עם חתימות מפורשות; בגרסה הראשונה החתימות חסרו והמדידה החזירה None אצל המשתמש)."""
    try:
        if sys.platform == "win32":
            try:
                import psutil
                mi = psutil.Process().memory_info()
                return getattr(mi, "peak_wset", mi.rss) / 2**20
            except ImportError:
                pass
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.WinDLL("psapi").GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            return pmc.PeakWorkingSetSize / 2**20 if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) else None
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 2**20 if sys.platform == "darwin" else v / 1024
    except Exception:  # noqa: BLE001 - מדידה אופציונלית
        return None


def config_id(size: int, strategy: str, min_scale: float, feather: int, overlap: int, blend: int, coarse: int = 0,
              coarse_size: int = 0, texture: int = 0) -> str:
    ms = f"-ms{min_scale:g}" if strategy == "auto" else ""
    return (f"s{size}-{strategy}{ms}-f{feather}-o{overlap}-b{blend}" + ("-cf" if coarse else "")
            + (f"{coarse_size}" if coarse and coarse_size else "") + ("-tx" if texture else ""))


def build_grid(sizes, strategies, min_scales, feathers, overlaps, blends=(32,), coarses=(0,), coarse_sizes=(0,),
               textures=(0,)) -> list[dict]:
    """רשימת תצורות (מילונים). min_scale רלוונטי רק ל-auto; overlap מצומצם בהמשך לפי התכנון."""
    out, seen = [], set()
    for size, strat, feather in itertools.product(sizes, strategies, feathers):
        for ms in (min_scales if strat == "auto" else [min_scales[0] if min_scales else 0.5]):
            for ov, bl, cf, cs, tx in itertools.product(overlaps, blends, coarses, coarse_sizes, textures):
                if not cf and cs:
                    continue                    # coarse_size רלוונטי רק עם מגס לדק
                cid = config_id(size, strat, ms, feather, ov, bl, cf, cs, tx)
                if cid not in seen:
                    seen.add(cid)
                    out.append(dict(id=cid, size=size, large_strategy=strat, min_scale=ms, feather_px=feather,
                                    tile_overlap=ov, tile_blend_px=bl, coarse_fine=int(cf), coarse_size=int(cs),
                                    texture_restore=int(tx)))
    return out


def plan_signature(rep, ov: int, blend: int = 0) -> tuple:
    """חתימת תכנון: שתי תצורות עם אותה חתימה (ואותו feather) נותנות אותה ריצה. החפיפה נכללת רק כשיש אריחים."""
    has_tile = any(g.kind == "tile" for g in rep.groups)
    return (rep.size, tuple((g.kind, round(g.scale, 3), g.windows) for g in rep.groups), (ov, blend) if has_tile else None)


def crop_box(mask: np.ndarray, margin: float = 0.15) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    my, mx = int((ys.max() - ys.min() + 1) * margin) + 8, int((xs.max() - xs.min() + 1) * margin) + 8
    return slice(max(0, ys.min() - my), min(h, ys.max() + 1 + my)), slice(max(0, xs.min() - mx), min(w, xs.max() + 1 + mx))


def to_ref_size(img: np.ndarray, mask: np.ndarray, ref_hw: tuple[int, int]):
    h, w = ref_hw
    return (cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
            (cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8))


def write_png(path: Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"קידוד PNG נכשל: {path}")
    path.write_bytes(buf.tobytes())


def safe_measure(img: np.ndarray, mask: np.ndarray) -> dict | None:
    try:
        return measure(img, mask).as_dict()
    except FixPicError as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- ריצה

def run_bench(model, case_dir: str | Path, configs: list[dict], out_dir: str | Path, *, base: PipelineConfig | None = None,
              max_runs: int = 80, save_full: bool = False, log=print) -> dict:
    """מריץ את התצורות על המקרה ושומר תוצאות. `model` = InpaintModel טעון (או מדומה בבדיקות). מחזיר את מילון התוצאות."""
    case_dir = Path(case_dir)
    case = load_case(case_dir)
    img, mask = case["rgb"], case["mask"]
    brush = (case["meta"].get("params") or {}).get("brush_px")
    name = case_dir.name
    out = Path(out_dir) / name
    (out / "crops").mkdir(parents=True, exist_ok=True)
    base = base or PipelineConfig()
    h, w = mask.shape
    cy, cx = crop_box(mask)
    ref = case["reference"]
    ref_hw = ref.shape[:2] if ref is not None and ref.shape[:2] != (h, w) else None

    rows: list[dict] = []
    skipped: list[dict] = []
    seen: dict[tuple, str] = {}
    cells: list[tuple[str, np.ndarray]] = [("original", img[cy, cx])]
    if ref is not None:
        rimg, rmask = (ref, to_ref_size(ref, mask, ref.shape[:2])[1]) if ref_hw else (ref, mask > 0)
        row = dict(id="reference", reference=True, **{f"ref_{k}": v for k, v in (safe_measure(rimg, rmask) or {}).items()})
        rows.append(row)
        # חיתוך הייחוס מוגדל לגודל המקור, כדי שהגיליון יציג את אותו אזור
        big = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC) if ref_hw else ref
        cells.append(("reference", big[cy, cx]))

    for i, cfg in enumerate(configs, 1):
        pc = replace(base, sizes=tuple(sorted(set(base.sizes) | {cfg["size"]} | ({cfg["coarse_size"]} if cfg.get("coarse_size") else set()))), large_strategy=cfg["large_strategy"],
                     min_scale=cfg["min_scale"], feather_px=cfg["feather_px"], tile_overlap=cfg["tile_overlap"],
                     tile_blend_px=cfg["tile_blend_px"], coarse_fine=bool(cfg.get("coarse_fine", 0)), coarse_belt_px=cfg.get("coarse_belt_px", 64), coarse_size=cfg.get("coarse_size") or None,
                     texture_restore=bool(cfg.get("texture_restore", 0)))
        pipe = Pipeline(model, pc)
        rep = pipe.plan(mask, cfg["size"], brush)
        sig = (plan_signature(rep, cfg["tile_overlap"], cfg["tile_blend_px"]), cfg["feather_px"], cfg.get("coarse_fine", 0), cfg.get("coarse_size", 0),
               cfg.get("texture_restore", 0))
        if sig in seen:
            skipped.append(dict(id=cfg["id"], reason=f"תכנון זהה ל-{seen[sig]}"))
            log(f"[{i}/{len(configs)}] {cfg['id']}: דילוג (תכנון זהה ל-{seen[sig]})")
            continue
        if rep.runs > max_runs:
            skipped.append(dict(id=cfg["id"], reason=f"{rep.runs} הרצות מעל --max-runs={max_runs}"))
            log(f"[{i}/{len(configs)}] {cfg['id']}: דילוג ({rep.runs} הרצות > {max_runs})")
            continue
        seen[sig] = cfg["id"]
        row = dict(cfg, planned_runs=rep.runs, kinds=",".join(sorted({g.kind for g in rep.groups})),
                   scale=round(rep.min_scale, 3))
        coarse_cell: list = []
        if cfg.get("coarse_fine"):          # אבחון M3.7.7: איך נראה המילוי אחרי המעבר הגס בלבד
            def _hook(box, crop, _img=img, _cell=coarse_cell):
                x0, y0, x1, y1 = box
                canvas = _img.copy()
                canvas[y0:y1, x0:x1] = crop
                _cell.append(canvas[cy, cx].copy())
            pipe.coarse_hook = _hook
        t0 = time.perf_counter()
        try:
            res = pipe.run(img, mask, cfg["size"], brush_px=brush)
        except FixPicError as e:
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            log(f"[{i}/{len(configs)}] {cfg['id']}: כשל {row['error']}")
            continue
        row["seconds"] = round(time.perf_counter() - t0, 2)
        row["model_calls"] = res.model_calls
        row["texture_groups"] = res.texture_groups
        row["peak_rss_mb"] = None if (v := peak_rss_mb()) is None else round(v)
        m = res.effective_mask
        row.update(safe_measure(res.image, m) or {})
        lines = pipe.tile_seams(mask, cfg["size"], brush)
        row["seam_lines"] = len(lines)
        try:
            row["seam_ratio"] = round(seam_ratio(res.image, lines), 3) if lines else None
        except FixPicError:
            row["seam_ratio"] = None
        if ref_hw:
            rimg, rmask = to_ref_size(res.image, m, ref_hw)
            row.update({f"ref_{k}": v for k, v in (safe_measure(rimg, rmask) or {}).items()})
        write_png(out / "crops" / f"{cfg['id']}.png", res.image[cy, cx])
        if coarse_cell:
            write_png(out / "crops" / f"{cfg['id']}-coarse.png", coarse_cell[-1])
        if save_full:
            write_png(out / f"{cfg['id']}.png", res.image)
        cells.append((cfg["id"], res.image[cy, cx]))
        rows.append(row)
        log(f"[{i}/{len(configs)}] {cfg['id']}: {row['seconds']}s, {res.model_calls} הרצות, "
            f"texture={row.get('texture_ratio', float('nan')):.2f} stripe={row.get('stripe_score', float('nan')):.1f} "
            f"cdiff={row.get('boundary_color_diff', float('nan')):.1f} seam={row['seam_ratio']}")

    data = dict(case=name, size=[w, h], mask_pixels=int((mask > 0).sum()), reference_size=list(ref.shape[1::-1]) if ref is not None else None,
                max_runs=max_runs, rows=rows, skipped=skipped)
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out / "results.csv", rows)
    (out / "summary.txt").write_text(summary_text(data), encoding="utf-8")
    write_png_sheet(out / "sheet.jpg", cells)
    return data


COLS = ["id", "planned_runs", "kinds", "scale", "seconds", "texture_groups", "peak_rss_mb", "texture_ratio", "stripe_score", "seam_ratio",
        "boundary_color_diff", "boundary_color_diff_p95", "ref_texture_ratio", "ref_stripe_score",
        "ref_boundary_color_diff", "error"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel קורא עברית/סימנים נכון
        wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def _f(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def summary_text(data: dict) -> str:
    L = [f"מקרה {data['case']}: {data['size'][0]}x{data['size'][1]}, מסכה {data['mask_pixels']:,} px"
         + (f", ייחוס {data['reference_size'][0]}x{data['reference_size'][1]}" if data["reference_size"] else "")]
    L.append("קריאה: texture מתחת ל-0.5 = מרוח, מעל 0.85 = תקין; seam מעל 1.3 = תפר אריחים; stripe הוא יחסי בלבד "
             "(פי 3 מהנמוך באותו מקרה = חשד); cdiff משקף בעיקר את הסצנה. פרטים: analyze_bench.py")
    L.append(f"{'id':28} {'runs':>5} {'kinds':>9} {'scale':>6} {'sec':>7} {'rssMB':>6} {'tex':>6} {'strp':>6} {'seam':>6} "
             f"{'cdiff':>6} | {'r.tex':>6} {'r.strp':>6} {'r.cdif':>6}")
    for r in data["rows"]:
        if r.get("error") and not r.get("reference"):
            L.append(f"{r['id']:28} כשל: {r['error']}")
            continue
        if r.get("reference"):
            L.append(f"{'reference':28} {'':>5} {'':>9} {'':>6} {'':>7} {'':>6} {'':>6} {'':>6} {'':>6} {'':>6} | "
                     f"{_f(r.get('ref_texture_ratio')):>6} {_f(r.get('ref_stripe_score'), 1):>6} {_f(r.get('ref_boundary_color_diff'), 1):>6}")
            continue
        L.append(f"{r['id']:28} {r['planned_runs']:>5} {r['kinds']:>9} {r['scale']:>6} {_f(r.get('seconds'), 1):>7} "
                 f"{_f(r.get('peak_rss_mb'), 0):>6} {_f(r.get('texture_ratio')):>6} {_f(r.get('stripe_score'), 1):>6} "
                 f"{_f(r.get('seam_ratio')):>6} {_f(r.get('boundary_color_diff'), 1):>6} | {_f(r.get('ref_texture_ratio')):>6} "
                 f"{_f(r.get('ref_stripe_score'), 1):>6} {_f(r.get('ref_boundary_color_diff'), 1):>6}")
    for s in data["skipped"]:
        L.append(f"דילוג {s['id']}: {s['reason']}")
    return "\n".join(L) + "\n"


def write_png_sheet(path: Path, cells: list[tuple[str, np.ndarray]], cols: int = 4, cell_w: int = 420) -> None:
    """גיליון השוואה: חיתוך לכל תצורה, מוקטן לרוחב אחד, עם תווית ASCII (cv2 אינו מצייר עברית)."""
    thumbs = []
    for label, im in cells:
        s = cell_w / im.shape[1]
        t = cv2.resize(im, (cell_w, max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        thumbs.append((label, t))
    ch = max(t.shape[0] for _, t in thumbs) + 22
    rows = -(-len(thumbs) // cols)
    sheet = np.full((rows * ch, cols * cell_w, 3), 255, np.uint8)
    for i, (label, t) in enumerate(thumbs):
        y, x = (i // cols) * ch, (i % cols) * cell_w
        sheet[y + 22:y + 22 + t.shape[0], x:x + cell_w] = t
        cv2.putText(sheet, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("קידוד הגיליון נכשל")
    path.write_bytes(buf.tobytes())


# ---------------------------------------------------------------- CLI

def _list(s: str, typ):
    return [typ(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="+", help="תיקיות מקרה (case.json + mask.png)")
    ap.add_argument("--out", default="m3_bench")
    ap.add_argument("--quick", action="store_true", help="רשת קטנה: bucket 384, auto/tile")
    ap.add_argument("--heavy", action="store_true", help="M3.7.5: חלון יחיד ב-768/1024/1536 (הקטנה בהרצה אחת)")
    ap.add_argument("--sizes"), ap.add_argument("--strategies"), ap.add_argument("--min-scales")
    ap.add_argument("--feathers"), ap.add_argument("--overlaps"), ap.add_argument("--blends"), ap.add_argument("--coarses", help="0,1: מגס לדק (M3.7.7) כבוי/דולק"), ap.add_argument("--textures", help="0,1: שחזור טקסטורה (M3.7.8) כבוי/דולק"), ap.add_argument("--coarse-sizes", dest="coarse_sizes", help="bucket של המעבר הגס, למשל 0,768,1024 (0 = כמו האריחים; רק עם --coarses 1)")
    ap.add_argument("--belt-px", dest="belt_px", type=int, default=64, help="רוחב חגורת המעבר העדין בתצורות מגס-לדק (--coarses 1); 0 = חלון גס יחיד בלי חגורה, כמו ברירת המחדל של ADR-012 (M3.7.9)")
    ap.add_argument("--max-runs", type=int, default=80)
    ap.add_argument("--plan-only", action="store_true", help="רק תכנון (בלי מודל)")
    ap.add_argument("--save-full", action="store_true")
    ap.add_argument("--model", default="big-lama")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()

    g = dict(HEAVY_GRID if a.heavy else QUICK_GRID if a.quick else DEFAULT_GRID)
    for key, typ in (("sizes", int), ("strategies", str), ("min_scales", float), ("feathers", int), ("overlaps", int), ("blends", int), ("coarses", int), ("coarse_sizes", int), ("textures", int)):
        v = getattr(a, key)
        if v:
            g[key] = _list(v, typ)
    configs = build_grid(**g)
    for c in configs:
        if c.get("coarse_fine"):
            c["coarse_belt_px"] = a.belt_px

    if a.plan_only:
        from fixpic.models.base import ModelInfo

        class _Stub:
            info = ModelInfo(name="stub", multiple=8, min_size=64, max_size=4096)

        for c in a.cases:
            case = load_case(c)
            brush = (case["meta"].get("params") or {}).get("brush_px")
            print(f"== {Path(c).name}: {case['mask'].shape[1]}x{case['mask'].shape[0]}, {len(configs)} תצורות ברשת")
            seen = {}
            for cfg in configs:
                pc = PipelineConfig(sizes=tuple(sorted({256, 384, 512, cfg["size"]})), large_strategy=cfg["large_strategy"],
                                    min_scale=cfg["min_scale"], feather_px=cfg["feather_px"], tile_overlap=cfg["tile_overlap"],
                                    tile_blend_px=cfg["tile_blend_px"], coarse_fine=bool(cfg.get("coarse_fine", 0)), coarse_belt_px=cfg.get("coarse_belt_px", 64), coarse_size=cfg.get("coarse_size") or None,
                     texture_restore=bool(cfg.get("texture_restore", 0)))
                rep = Pipeline(_Stub(), pc).plan(case["mask"], cfg["size"], brush)
                sig = (plan_signature(rep, cfg["tile_overlap"], cfg["tile_blend_px"]), cfg["feather_px"], cfg.get("coarse_fine", 0), cfg.get("coarse_size", 0),
               cfg.get("texture_restore", 0))
                note = f"= {seen[sig]}" if sig in seen else (f"> max-runs ({a.max_runs})" if rep.runs > a.max_runs else "תרוץ")
                seen.setdefault(sig, cfg["id"])
                print(f"  {cfg['id']:28} {rep.runs:>4} הרצות, קנה מידה {rep.min_scale:.3f}, {note}")
        return

    from fixpic.cli import DEFAULT_MODELS
    from fixpic.models import ModelManager

    model = ModelManager(a.models_dir or DEFAULT_MODELS).load(a.model, threads=a.threads)
    for c in a.cases:
        print(f"== {c}")
        data = run_bench(model, c, configs, a.out, max_runs=a.max_runs, save_full=a.save_full)
        print(summary_text(data))
        print(f"נשמר ב-{Path(a.out) / Path(c).name} (sheet.jpg, results.csv, summary.txt)")


if __name__ == "__main__":
    main()
