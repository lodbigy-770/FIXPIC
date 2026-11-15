"""M3.7.5: האם Big LaMa (דרך אותו backend של היישום) מקבל חלונות גדולים, וכמה זמן וזיכרון הם עולים.

הרצה אצל המשתמש (Windows, אותו מחשב כמו M1.5):
  python tools/m3/big_window_check.py                      # 512 768 1024
  python tools/m3/big_window_check.py --sizes 512 768 1024 1536 --threads 3 --repeats 2
  python tools/m3/big_window_check.py --out big_window.json

לכל גודל (בסדר עולה, כי שיא הזיכרון של התהליך עולה בלבד): טעינה/compile (`warmup`) בנפרד, הרצה ראשונה, חציון של
חזרות, ושיא זיכרון אחרי הגודל. נבדק גם שפלט המודל בצורה הנכונה ושמחוץ למסכה אין שינוי גדול (בדיקת שפיות). כשל בגודל
(חריגת זיכרון או צורה לא נתמכת) נרשם והריצה ממשיכה ונעצרת בגודל הבא. מחזיר JSON; את הקובץ יש להחזיר.

הנתונים שכבר קיימים (ORT, `latency_results.json`, 4 threads): 768: 9.7s, 1024: 17.9s, שיא תהליך 2.05GB."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from quality_bench import peak_rss_mb  # noqa: E402


def check_size(model, size: int, repeats: int) -> dict:
    rng = np.random.default_rng(size)
    img = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    q = size // 4
    mask[q:3 * q, q:3 * q] = 1
    row: dict = {"size": size}
    warm = getattr(model, "warmup", None)
    if warm:
        t = time.perf_counter()
        warm([size], None)
        row["warmup_s"] = round(time.perf_counter() - t, 2)
    times = []
    for i in range(repeats + 1):
        t = time.perf_counter()
        out = model.run(img, mask, None)
        times.append(time.perf_counter() - t)
        if i == 0:
            row["first_s"] = round(times[0], 2)
            row["shape_ok"] = tuple(out.shape) == (size, size, 3) and out.dtype == np.uint8
            outside = np.abs(out.astype(int) - img.astype(int))[mask == 0]
            row["outside_mean_abs"] = round(float(outside.mean()), 2)   # בדיקת שפיות בלבד (המודל עשוי לשנות מעט)
    row["median_s"] = round(statistics.median(times[1:] or times), 2)
    v = peak_rss_mb()
    row["peak_rss_mb"] = None if v is None else round(v)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="big-lama")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("big_window.json"))
    a = ap.parse_args()

    from fixpic.cli import DEFAULT_MODELS
    from fixpic.models import ModelManager

    t = time.perf_counter()
    model = ModelManager(a.models_dir or DEFAULT_MODELS).load(a.model, threads=a.threads)
    rows = [{"load_s": round(time.perf_counter() - t, 2), "backend": type(model).__name__}]
    print(rows[0])
    for size in sorted(a.sizes):
        try:
            row = check_size(model, size, a.repeats)
        except Exception as e:  # noqa: BLE001 - כשל בגודל הוא תוצאה
            row = {"size": size, "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            print(row)
            break
        rows.append(row)
        print(row)
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("נשמר:", a.out)


if __name__ == "__main__":
    main()
