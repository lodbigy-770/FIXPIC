"""M3.7.2: מה ה-Pipeline מתכנן למסכה נתונה, בלי מודל ובלי תמונה: הרחבה, קבוצות, חלון, קנה מידה, אסטרטגיה, הרצות מודל.

הרצה: `python tools/m3/large_region_diag.py CASE_DIR|MASK.png [...] [--sizes 256,384,512,768,1024]
[--strategy auto|tile|downscale] [--min-scale 0.5] [--max-tiles 64] [--brush PX]`
ברירת המחדל של הממשק היא `default_size` (384). `scale` < 1 = הפיקסלים נשלחים למודל מוקטנים (1/scale פעמים),
והתוצאה מוגדלת חזרה; זו הסיבה הסבירה ל"צל מטושטש" באזורים גדולים."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from fixpic.casefile import read_mask  # noqa: E402
from fixpic.models.base import ModelInfo  # noqa: E402
from fixpic.pipeline import Pipeline, PipelineConfig  # noqa: E402


class _Stub:
    info = ModelInfo(name="stub", multiple=8, min_size=64, max_size=4096)


def load_mask(arg: str) -> tuple[str, np.ndarray, float | None]:
    p = Path(arg)
    if p.is_dir():
        meta = json.loads((p / "case.json").read_text(encoding="utf-8"))
        return p.name, read_mask(p / "mask.png"), (meta.get("params") or {}).get("brush_px")
    return p.stem, read_mask(p), None


def report(name: str, mask: np.ndarray, brush: float | None, cfg: PipelineConfig, sizes: list[int]) -> list[str]:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    lines = [f"== {name}: {w}x{h} ({w * h / 1e6:.1f}MP), מסכה {int((mask > 0).sum()):,} px ({100 * (mask > 0).mean():.1f}%), "
             f"תיבה {xs.max() - xs.min() + 1}x{ys.max() - ys.min() + 1}"]
    pipe = Pipeline(_Stub(), replace(cfg, sizes=tuple(sorted(set(cfg.sizes) | set(sizes)))))
    lines.append(f"{'bucket':>7} {'הרחבה':>6} {'קבוצות':>7} {'אסטרטגיה':>10} {'קנה מידה':>9} {'הקטנה פי':>9} {'הרצות':>6}")
    for s in sizes:
        rep = pipe.plan(mask, s, brush)
        kinds = ",".join(sorted({g.kind for g in rep.groups}))
        sc = rep.min_scale
        lines.append(f"{s:>7} {rep.dilate_px:>6} {len(rep.groups):>7} {kinds:>10} {sc:>9.3f} {1 / sc:>9.1f} {rep.runs:>6}"
                     f"{'   <- ברירת המחדל' if s == cfg.default_size else ''}")
    d = pipe.plan(mask, cfg.default_size, brush)
    for i, g in enumerate(d.groups[:8]):
        lines.append(f"  קבוצה {i}: תיבה {g.box} ({g.box[2] - g.box[0]}x{g.box[3] - g.box[1]}), {g.mask_px:,} px, {g.kind}, "
                     f"חלון {g.window[0]}x{g.window[1]}, קנה מידה {g.scale:.3f}, {g.windows} הרצות")
    if len(d.groups) > 8:
        lines.append(f"  ... ועוד {len(d.groups) - 8} קבוצות")
    if d.min_scale < 0.5:
        lines.append(f"  !! בברירת המחדל הפיקסלים נשלחים למודל מוקטנים פי {1 / d.min_scale:.1f}: צפוי טשטוש")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--sizes", default="256,384,512,768,1024")
    ap.add_argument("--strategy", choices=("auto", "tile", "downscale"))
    ap.add_argument("--min-scale", type=float)
    ap.add_argument("--max-tiles", type=int)
    ap.add_argument("--brush", type=float, help="עקיפת קוטר המברשת (פיקסלי תמונה)")
    a = ap.parse_args()
    cfg = PipelineConfig()
    over = {k: v for k, v in (("large_strategy", a.strategy), ("min_scale", a.min_scale), ("max_tiles", a.max_tiles)) if v is not None}
    cfg = replace(cfg, **over)
    sizes = [int(x) for x in a.sizes.split(",")]
    for c in a.cases:
        name, mask, brush = load_mask(c)
        print("\n".join(report(name, mask, a.brush if a.brush is not None else brush, cfg, sizes)), "\n")


if __name__ == "__main__":
    main()
