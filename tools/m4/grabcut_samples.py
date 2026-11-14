"""מדידת `GrabCutProvider` על `samples/` (AUTO-SELECT.md §7 שלב 3): זמן חם ותמונות-שכבה לבדיקה ויזואלית.

הרצה מהשורש:  python tools/m4/grabcut_samples.py [--out DIR] [--repeat N]
אין כאן אמת-קרקע: הקופסאות והנקודות נבחרו ידנית לפי התמונות, והמסקנה על איכות היא **בדיקה בעין** של ה-PNG-ים.
כל מקרה כותב `<out>/<שם>.png` (המסכה באדום, קו המתאר בצהוב, הקלט בציאן).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixpic.select import Point, Prompts            # noqa: E402
from fixpic.select.grabcut import GrabCutProvider   # noqa: E402

# (שם, תמונה, Prompts) – קואורדינטות בפיקסלי התמונה המלאה
CASES = [
    ("person_box", "person", Prompts(box=(160, 285, 500, 668))),
    ("person_box_loose", "person", Prompts(box=(100, 230, 534, 720))),
    ("person_box_loose_scribble", "person",       # אותו מלבן רפוי + משיכה שלילית (נקודות צפופות) על הספסל והפרחים
     Prompts(box=(100, 230, 534, 720),
             points=tuple(Point(x, y, False) for y in (290, 335, 390) for x in range(330, 530, 14)))),
    ("person_lasso", "person", Prompts(polygon=((165, 280), (330, 280), (330, 400), (420, 500), (500, 600), (500, 670),
                                                 (300, 670), (160, 620), (170, 450)))),
    ("bin_box", "text", Prompts(box=(0, 285, 90, 412))),
    ("sign_box", "text", Prompts(box=(252, 97, 542, 158))),
    ("signs_strip_box", "text", Prompts(box=(258, 412, 562, 449))),
    ("one_sign_in_strip_box", "text", Prompts(box=(258, 412, 295, 449))),
    ("booth_box", "text", Prompts(box=(215, 75, 580, 490))),
]


def overlay(bgr: np.ndarray, rect, window, prompts: Prompts) -> np.ndarray:
    m = np.zeros(bgr.shape[:2], np.uint8)
    x0, y0, x1, y1 = rect
    m[y0:y1, x0:x1] = window
    out = bgr.copy()
    out[m > 0] = (0.5 * out[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, (0, 255, 255), 1)
    if prompts.box:
        cv2.rectangle(out, prompts.box[:2], prompts.box[2:], (255, 255, 0), 1)
    if prompts.polygon:
        cv2.polylines(out, [np.round(np.array(prompts.polygon)).astype(np.int32)], True, (255, 255, 0), 1)
    for p in prompts.points:
        cv2.circle(out, (int(p.x), int(p.y)), 4, (0, 255, 0) if p.positive else (0, 0, 255), -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "m4_out"))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--roi-px", type=int, default=270_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = GrabCutProvider(iterations=a.iterations, max_roi_px=a.roi_px)
    ctxs: dict[str, tuple[np.ndarray, object]] = {}
    print(f"{'מקרה':26} {'זמן חם (ms)':>12} {'ROI':>10} {'פיקסלים':>9}  מלבן")
    for name, img, prompts in CASES:
        if img not in ctxs:
            bgr = cv2.imdecode(np.fromfile(str(ROOT / "samples" / f"{img}.png"), np.uint8), cv2.IMREAD_COLOR)
            ctxs[img] = (bgr, prov.prepare(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        bgr, ctx = ctxs[img]
        times, prop = [], None
        for _ in range(a.repeat + 1):                 # הראשונה = חימום, לא נספרת
            t = time.perf_counter()
            prop = prov.propose(ctx, prompts)
            times.append((time.perf_counter() - t) * 1000)
        ms = statistics.median(times[1:])
        if prop is None:
            print(f"{name:26} {ms:12.0f} {'-':>10} {0:9d}  (ריק)")
            continue
        cv2.imencode(".png", overlay(bgr, prop.rect, prop.window, prompts))[1].tofile(str(out / f"{name}.png"))
        x0, y0, x1, y1 = prop.rect
        print(f"{name:26} {ms:12.0f} {x1 - x0:>5}x{y1 - y0:<4} {prop.pixel_count:9d}  {prop.rect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
