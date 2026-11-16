"""M5 – בדיקת ביצועים על חומרת יעד: Latency חם של `Pipeline.run` מול תקציב ה-Latency של M1.5.

שימוש (על i5-6500T, מחובר לחשמל, בלי תוכנות כבדות ברקע):
  python tools/m5/latency_budget.py --model big-lama --threads 4 --json latency_budget.json --md latency_budget.md
  python tools/m5/latency_budget.py --model big-lama --threads 3        # ברירת המחדל של המוצר
  python tools/m5/latency_budget.py                                     # מודל סינתטי: בדיקת צנרת בלבד, בלי פסק דין

התקציב (PROGRESS, הוכרע 2026-11-12, ב-4 threads על i5-6500T): bucket 256 עד ~0.8s (אינטראקטיבי), 384 עד ~2s (רגיל, ברירת מחדל),
512 עד ~4s (איכות, ידני); 768 ומעלה ידני בלבד (מדווח בלי תקציב).

מה נמדד: `Pipeline.run` מקצה לקצה על תמונה סינתטית (ברירת מחדל 12MP) עם מסכה מרכזית (משיכה קטנה וגדולה יחסית), אחרי Warm-up.
כלומר כולל הכנת קלט, חיתוך, Inference והרכבה, ולא רק את המודל; לכן הסף כאן מחמיר יותר מהמדידה הקודמת של ה-Inference בלבד.
כל תרחיש: ריצה אחת נזרקת (חימום), ואחריה `--reps` ריצות; הפסק דין לפי החציון (וגם מדווח מקסימום).
Warm-up (compile) נמדד בנפרד ומדווח, כי הוא רץ ברקע במוצר ואינו חלק מה-Latency לפעולה.

עם מודל סינתטי הזמנים אינם משקפים Inference, ולכן אין פסק דין (`verdict` = None); המטרה בדיקת הצנרת.
קוד יציאה: 0 אם כל התרחישים עם תקציב עברו (או שאין פסק דין), 1 אם תרחיש חרג או נכשל."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# bucket -> (תקציב בשניות או None, תיאור)
BUDGETS: dict[int, tuple[float | None, str]] = {
    256: (0.8, "אינטראקטיבי"),
    384: (2.0, "רגיל"),
    512: (4.0, "איכות"),
    768: (None, "ידני (ללא תקציב)"),
}
# שם מסכה -> צלע הריבוע (בפיקסלים של התמונה)
MASKS: dict[str, int] = {"stroke": 96, "region": 400}


class SyntheticModel:
    """ממלא את המסכה בצבע קבוע; חוזה המודל בלבד (אין Inference אמיתי)."""

    def __init__(self):
        from fixpic.models import ModelInfo
        self.info = ModelInfo(name="synthetic")
        self.backend = "synthetic"

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        out = image.copy()
        out[mask > 0] = (120, 110, 100)
        return out


def make_image(w: int, h: int):
    """תמונה סינתטית דטרמיניסטית עם מרקם (מדרג + רעש), כדי שלא תהיה שטוחה לגמרי."""
    import numpy as np
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
    base = (x * 0.6 + y * 0.4) * np.ones((1, 1, 3), np.float32)
    noise = rng.integers(-12, 13, size=(h, w, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def make_mask(w: int, h: int, side: int):
    import numpy as np
    m = np.zeros((h, w), np.uint8)
    side = max(1, min(side, w, h))
    y0, x0 = (h - side) // 2, (w - side) // 2
    m[y0:y0 + side, x0:x0 + side] = 255
    return m


def summarize(times: list[float]) -> dict:
    return {"n": len(times), "min_s": round(min(times), 3), "median_s": round(statistics.median(times), 3), "max_s": round(max(times), 3)}


def verdict_for(median_s: float | None, budget_s: float | None, real_model: bool) -> bool | None:
    """None = אין פסק דין (מודל סינתטי, בלי תקציב, או אין מדידה)."""
    if not real_model or budget_s is None or median_s is None:
        return None
    return median_s <= budget_s


def measure_scenario(pipe, image, mask, size: int, reps: int) -> dict:
    """ריצה אחת נזרקת, ואז `reps` ריצות. כשל מוחזר כ-`ok=False` בלי להפיל את שאר התרחישים."""
    from fixpic.errors import FixPicError
    try:
        pipe.run(image, mask, size)
        times, last = [], None
        for _ in range(max(1, reps)):
            t = time.perf_counter()
            last = pipe.run(image, mask, size)
            times.append(time.perf_counter() - t)
        return {"ok": True, **summarize(times), "model_calls": last.model_calls, "size_used": last.size_used,
                "downscaled": bool(last.roi.downscaled), "scale": round(float(last.roi.scale), 3)}
    except (FixPicError, MemoryError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def run_benchmark(model, model_label: str, threads: int, mp: float, sizes: list[int], masks: list[str], reps: int,
                  warm: bool = True) -> dict:
    from fixpic.pipeline import Pipeline, PipelineConfig
    real = model_label != "synthetic"
    h = int((mp * 1e6 / 1.5) ** 0.5)
    w = int(h * 1.5)
    image = make_image(w, h)
    pipe = Pipeline(model, PipelineConfig())
    warm_s = None
    if warm:
        t = time.perf_counter()
        pipe.warmup(sizes=sorted(set(sizes)))
        warm_s = round(time.perf_counter() - t, 2)
    rows = []
    for size in sizes:
        budget, desc = BUDGETS.get(size, (None, "ללא תקציב"))
        for mname in masks:
            r = {"size": size, "mask": mname, "mask_side": MASKS[mname], "budget_s": budget, "class": desc}
            r.update(measure_scenario(pipe, image, make_mask(w, h, MASKS[mname]), size, reps))
            r["verdict"] = verdict_for(r.get("median_s"), budget, real) if r["ok"] else False
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    info = {"model": model_label, "backend": getattr(model, "backend", ""), "threads": threads, "image_mp": round(w * h / 1e6, 1),
            "image_wh": [w, h], "reps": reps, "warmup_s": warm_s, "platform": sys.platform, "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    return {"info": info, "rows": rows}


def to_markdown(result: dict) -> str:
    i = result["info"]
    lines = [f"# Latency מול תקציב (M1.5)", "",
             f"מודל: {i['model']} ({i['backend'] or 'ללא backend'}), threads {i['threads']}, תמונה {i['image_mp']}MP, {i['reps']} ריצות חמות לתרחיש, "
             f"Warm-up {i['warmup_s']}s, CPU {i['cpu_count']} ליבות, {i['platform']}/{i['python']}", "",
             "| bucket | סוג | מסכה | תקציב s | חציון s | מקס' s | הרצות מודל | מוקטן | פסק דין |", "|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        v = {True: "עבר", False: "חרג/נכשל", None: "-"}[r["verdict"]]
        if r["ok"]:
            lines.append(f"| {r['size']} | {r['class']} | {r['mask']} ({r['mask_side']}px) | {r['budget_s'] or '-'} | {r['median_s']} | {r['max_s']} | "
                         f"{r['model_calls']} | {'כן' if r['downscaled'] else 'לא'} | {v} |")
        else:
            lines.append(f"| {r['size']} | {r['class']} | {r['mask']} | {r['budget_s'] or '-'} | - | - | - | - | {v}: {r['error']} |")
    if i["model"] == "synthetic":
        lines += ["", "מודל סינתטי: הזמנים אינם Inference אמיתי, ואין פסק דין."]
    return "\n".join(lines)


def exit_code(result: dict) -> int:
    return 1 if any(r["verdict"] is False for r in result["rows"]) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="שם מודל אמיתי (למשל big-lama); ברירת מחדל: מודל סינתטי (בדיקת צנרת)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--mp", type=float, default=12.0, help="גודל התמונה הסינתטית במגה-פיקסל")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--masks", nargs="+", choices=sorted(MASKS), default=["stroke", "region"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    if a.model:
        from fixpic.cli import DEFAULT_MODELS
        from fixpic.models.manager import ModelManager
        model, label = ModelManager(DEFAULT_MODELS).load(a.model, threads=a.threads), a.model
    else:
        model, label = SyntheticModel(), "synthetic"
    result = run_benchmark(model, label, a.threads, a.mp, a.sizes, a.masks, a.reps, warm=not a.no_warmup)
    md = to_markdown(result)
    print("\n" + md)
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
