"""כיוון `MagicWandProvider` על `samples/` (AUTO-SELECT.md §7 שלב 2, "לא נבדק" ברמה 0): סוויפ סבילות, דיוק/דליפה/כיסוי, יציבות ללחיצה וזמן.

הרצה מהשורש:  python tools/m4/wand_samples.py [--out DIR] [--tols 4,8,...] [--overlay-tols 20,30] [--repeat N]

אמת-הקרקע כאן **גסה ונסמנה ידנית** (מצולעים לפי רשת על התמונות), ולכן המספרים הם אינדיקציה ולא מדד מוחלט:
- `poly`/`holes` – האזור שהלחיצה אמורה לבחור (פחות חורים). שוליים של `margin` פיקסלים סביב כל גבול אינם נספרים לכאן ולכאן.
- `ignore` – מלבנים שאינם נספרים כלל (טקסט ושלטים על הדלפק).
- `recall=False` – המצולע רק חוסם (מגבלה עליונה): נמדדת דליפה החוצה אבל לא כיסוי, כי הבחירה לא אמורה למלא אותו כולו (צל מנוקד, שמיים בין עלים).
מדדים לכל (מקרה, סבילות): `pix` פיקסלים שנבחרו, `leak` חלקם מחוץ לאזור, `rec` כיסוי האזור (אם `recall`), `ms` זמן חם.
יציבות: 8 לחיצות במרחק 5 פיקסלים מהמרכז (בתוך האזור), IoU חציוני ומינימלי מול הבחירה מהמרכז.
`--overlay-tols` כותב לכל מקרה `<out>/<שם>_t<סבילות>.jpg` (JPEG, לבדיקה בעין בלבד) (מסכה באדום, מתאר בצהוב, האזור בציאן, הלחיצה בירוק).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixpic.select import Point, Prompts            # noqa: E402
from fixpic.select.wand import MagicWandProvider    # noqa: E402

BABY = [(180, 300), (250, 288), (305, 320), (310, 380), (300, 415), (380, 490), (420, 520), (440, 560), (440, 620),
        (495, 615), (495, 632), (380, 667), (305, 667), (260, 642), (200, 627), (162, 598), (195, 540), (205, 470),
        (185, 400)]


@dataclass
class Case:
    name: str
    kind: str                                   # sky / wood / grass / shadow
    img: str
    seed: tuple[int, int]
    poly: list[tuple[int, int]]
    holes: list[list[tuple[int, int]]] = field(default_factory=list)
    ignore: list[tuple[int, int, int, int]] = field(default_factory=list)
    recall: bool = True
    margin: int = 8


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


CASES = [
    Case("sky_person_left", "sky", "person", (30, 40), [(12, 0), (60, 0), (85, 62), (62, 86), (12, 86)], recall=False),
    Case("sky_person_right", "sky", "person", (310, 70), rect(268, 34, 352, 90), recall=False),
    Case("sky_text", "sky", "text", (410, 8), rect(385, 0, 450, 25), recall=False),
    Case("wood_back_wall", "wood", "text", (400, 230), rect(278, 160, 528, 242), margin=6),
    Case("wood_counter_front", "wood", "text", (400, 462), [(236, 354), (565, 354), (562, 470), (238, 472)],
         ignore=[(258, 411, 562, 454), (332, 381, 468, 406)], margin=6),
    Case("grass_left", "grass", "person", (100, 600), rect(0, 370, 534, 800), holes=[BABY], margin=10),
    Case("grass_right", "grass", "person", (450, 740), rect(0, 370, 534, 800), holes=[BABY], margin=10),
    Case("shadow_ground_a", "shadow", "text", (700, 430), rect(570, 384, 800, 500), recall=False),
    Case("shadow_ground_b", "shadow", "text", (620, 420), rect(570, 384, 800, 500), recall=False),
    Case("pink_outfit", "pink", "person", (330, 470),
         [(208, 445), (225, 418), (290, 410), (330, 420), (372, 462), (412, 512), (437, 552), (437, 605), (380, 625),
          (335, 598), (262, 606), (250, 572), (222, 548), (212, 490)], recall=False, margin=10),
]

JITTER = [(5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4), (-4, -4)]


def fill(shape, polys, value=255):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], value)
    return m


class Truth:
    def __init__(self, case: Case, shape):
        g = fill(shape, [case.poly])
        for h in case.holes:
            cv2.fillPoly(g, [np.array(h, np.int32)], 0)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * case.margin + 1,) * 2)
        self.g = g > 0
        self.dil = cv2.dilate(g, k) > 0
        self.ero = cv2.erode(g, k) > 0
        ign = np.zeros(shape, bool)
        for x0, y0, x1, y1 in case.ignore:
            ign[y0:y1, x0:x1] = True
        self.ign = ign
        self.case = case

    def full(self, prop, shape) -> np.ndarray:
        m = np.zeros(shape, bool)
        if prop is not None:
            x0, y0, x1, y1 = prop.rect
            m[y0:y1, x0:x1] = prop.window > 0
        return m

    def score(self, s: np.ndarray):
        s = s & ~self.ign
        n = int(s.sum())
        leak = float((s & ~self.dil).sum()) / n if n else 0.0
        e = self.ero & ~self.ign
        rec = float((s & e).sum()) / max(1, int(e.sum())) if self.case.recall else None
        return n, leak, rec


def load(img):
    bgr = cv2.imdecode(np.fromfile(str(ROOT / "samples" / f"{img}.png"), np.uint8), cv2.IMREAD_COLOR)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(rgb: np.ndarray, variant: str) -> np.ndarray:
    """מדמה עיבוד מקדים בתוך `prepare` (בתמונות עד 1.5MP זהה, כי אין הקטנה לפניו)."""
    if variant == "plain":
        return rgb
    if variant.startswith("g"):
        return cv2.GaussianBlur(rgb, (0, 0), float(variant[1:]))
    if variant.startswith("m"):
        return cv2.medianBlur(rgb, int(variant[1:]))
    raise SystemExit(f"variant לא מוכר: {variant}")


def overlay(bgr, s, truth: Truth, seed):
    out = bgr.copy()
    out[s] = (0.5 * out[s] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    cnt, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnt, -1, (0, 255, 255), 1)
    c = truth.case
    for p in [c.poly, *c.holes]:
        cv2.polylines(out, [np.array(p, np.int32)], True, (255, 255, 0), 1)
    cv2.circle(out, seed, 4, (0, 255, 0), -1)
    return out


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "m4_out" / "wand"))
    ap.add_argument("--tols", default="4,8,12,16,20,25,30,40,50")
    ap.add_argument("--overlay-tols", default="20,30")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--variant", default="plain",
                    help="עיבוד מקדים של תמונת הניתוח: plain (כמו היום) | g<σ> (טשטוש גאוסי, למשל g1.5) | m<k> (חציון k×k, למשל m5)")
    a = ap.parse_args()
    tols = [int(t) for t in a.tols.split(",")]
    ov_tols = {int(t) for t in a.overlay_tols.split(",") if t}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = MagicWandProvider()
    print(f"variant: {a.variant}")
    imgs: dict[str, tuple] = {}
    results: dict[str, dict[int, tuple]] = {}
    print(f"{'מקרה':20} " + " ".join(f"{t:>14}" for t in tols))
    print(f"{'':20} " + " ".join(f"{'pix leak% rec%':>14}" for _ in tols))
    for c in CASES:
        if c.img not in imgs:
            bgr, rgb = load(c.img)
            imgs[c.img] = (bgr, rgb, prov.prepare(preprocess(rgb, a.variant)))
        bgr, rgb, ctx = imgs[c.img]
        shape = rgb.shape[:2]
        tr = Truth(c, shape)
        row, results[c.name] = [], {}
        for t in tols:
            pr = Prompts(points=(Point(*c.seed),), tolerance=t)
            ts, prop = [], None
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                prop = prov.propose(ctx, pr)
                ts.append((time.perf_counter() - t0) * 1000)
            s = tr.full(prop, shape)
            n, leak, rec = tr.score(s)
            results[c.name][t] = (n, leak, rec, statistics.median(ts))
            row.append(f"{n:>6} {leak * 100:>3.0f} {('-' if rec is None else f'{rec * 100:.0f}'):>3}")
            if t in ov_tols:
                cv2.imencode(".jpg", overlay(bgr, s, tr, c.seed), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out / f"{c.name}_t{t}.jpg"))
        print(f"{c.name:20} " + " ".join(f"{r:>14}" for r in row))

    print("\nזמן חם (ms, חציוני) לפי סבילות: מקסימום על המקרים")
    print(" ".join(f"{t}:{max(results[c.name][t][3] for c in CASES):.0f}" for t in tols))

    print("\nיציבות ללחיצה (IoU חציוני / מינימלי מול הלחיצה במרכז, 8 לחיצות במרחק ~5px בתוך האזור):")
    print(f"{'מקרה':20} " + " ".join(f"{t:>9}" for t in tols))
    for c in CASES:
        bgr, rgb, ctx = imgs[c.img]
        shape = rgb.shape[:2]
        tr = Truth(c, shape)
        row = []
        for t in tols:
            centre = tr.full(prov.propose(ctx, Prompts(points=(Point(*c.seed),), tolerance=t)), shape)
            ious = []
            for dx, dy in JITTER:
                x, y = c.seed[0] + dx, c.seed[1] + dy
                if not (0 <= x < shape[1] and 0 <= y < shape[0]) or not tr.g[y, x] or tr.ign[y, x]:
                    continue
                s = tr.full(prov.propose(ctx, Prompts(points=(Point(x, y),), tolerance=t)), shape)
                ious.append(iou(centre, s))
            row.append("-" if not ious else f"{statistics.median(ious):.2f}/{min(ious):.2f}")
        print(f"{c.name:20} " + " ".join(f"{r:>9}" for r in row))

    print(f"\nפסק לפי סבילות: מספר מקרים עם דליפה ≤5% (מתוך {len(CASES)}); ובמקרים עם recall – ממוצע כיסוי")
    for t in tols:
        ok = sum(1 for c in CASES if results[c.name][t][1] <= 0.05)
        recs = [results[c.name][t][2] for c in CASES if results[c.name][t][2] is not None]
        print(f"  tol {t:>3}: {ok:>2}/{len(CASES)} בלי דליפה;  כיסוי ממוצע {statistics.mean(recs) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
