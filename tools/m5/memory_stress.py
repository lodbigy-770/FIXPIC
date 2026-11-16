"""M5 – תמונות גדולות ועומס זיכרון: טעינה -> הסרה (Pipeline) -> שמירה על תמונות סינתטיות בגדלים עולים, ושיא זיכרון (RSS) לכל שלב.

שימוש:
  python tools/m5/memory_stress.py                                  # 12,24,50,100,150 MP, מודל סינתטי, מסכה 10%
  python tools/m5/memory_stress.py --mps 12 24 --mask-fracs 0.1 0.4 --json out.json --md out.md
  python tools/m5/memory_stress.py --model big-lama --threads 3     # מודל אמיתי (דורש models/big-lama/inpainter.onnx אמיתי, לא קובץ LFS)

כל גודל רץ בתהליך נפרד (שיא הזיכרון מצטבר לאורך חיי תהליך), והתמונה נוצרת בתהליך נפרד לפני המדידה, כדי שהיצירה לא תנפח את המספרים.
גודל שההערכה שלו (`estimate_memory_mb` + התמונה והמסכה) חורגת מהתקציב (ברירת מחדל 70% מהזיכרון הפיזי) מדולג ומסומן, כדי לא להפיל את המכונה.

המודל הסינתטי ממלא את המסכה ומחזיר תמונה בגודל הקלט, כלומר **נמדד הזיכרון של הכנת הקלט, החיתוך, ההרכבה והשמירה, לא של המודל האמיתי**
(שצריך להוסיף לו `model_overhead_mb`). עם `--model` נמדד גם הוא. ההערכה `estimate_mb` היא מה שה-Pipeline מחשב לפני ריצה (תקרת זיכרון).
שיא ה-RSS נמדד לפי `ru_maxrss` (POSIX) או `PeakWorkingSetSize` (Windows; לא נבדק ב-Windows בסביבת הפיתוח)."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MB = 2**20
STAGES = ("load", "run", "save")


def peak_rss_mb() -> float:
    """שיא ה-RSS של התהליך הנוכחי (MB)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo נכשל")
        return pmc.PeakWorkingSetSize / MB
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (v if sys.platform == "darwin" else v * 1024) / MB          # Linux: KB; macOS: בתים


def physical_memory_mb() -> float:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / MB
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / MB


def dims_for_mp(mp: float, aspect: float = 1.5) -> tuple[int, int]:
    """(רוחב, גובה) ליחס 3:2 בכ-`mp` מגה-פיקסל."""
    w = int(math.sqrt(mp * 1e6 * aspect))
    return w, max(1, int(w / aspect))


def needed_mb(w: int, h: int, model_overhead_mb: int = 0) -> int:
    """הערכה גסה לשיא התהליך: אומדן ה-Pipeline + תמונה (3B/px) + מסכה (1B/px) + תמונת פלט (3B/px) שמוחזקות אצל המתקשר."""
    from fixpic.pipeline.pipeline import estimate_memory_mb
    return estimate_memory_mb(h, w, model_overhead_mb) + int(7 * w * h / MB) + 1


# ------------------------------------------------------------------------------------------------ תהליכי ילד
def gen_image(path: Path, w: int, h: int) -> None:
    """תמונה סינתטית (גרדיאנט + רעש עדין, דומה בדחיסה לתמונת JPEG אמיתית) בקובץ JPEG."""
    import numpy as np
    from PIL import Image
    img = Image.new("RGB", (w, h))
    rng = np.random.default_rng(0)
    strip = 512
    xs = np.linspace(0, 255, w, dtype=np.float32)
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        ys = np.linspace(y0 * 255 / h, y1 * 255 / h, y1 - y0, dtype=np.float32)[:, None]
        base = np.stack([xs[None, :] + 0 * ys, ys + 0 * xs[None, :], (xs[None, :] + ys) / 2], -1)
        noise = rng.normal(0, 6, base.shape).astype(np.float32)
        img.paste(Image.fromarray((base + noise).clip(0, 255).astype(np.uint8)), (0, y0))
    img.save(path, "JPEG", quality=85)


class SyntheticModel:
    """ממלא את המסכה בצבע קבוע; מחזיר תמונה בגודל הקלט (חוזה המודל)."""
    def __init__(self):
        from fixpic.models import ModelInfo
        self.info = ModelInfo(name="synthetic")

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        out = image.copy()
        out[mask > 0] = (120, 110, 100)
        return out


def measure_child(image: Path, mask_frac: float, size: int, model_name: str | None, threads: int, out_dir: Path) -> dict:
    import numpy as np
    from fixpic.errors import FixPicError
    from fixpic.imgio import load_image, save_image
    from fixpic.pipeline import Pipeline, PipelineConfig
    from fixpic.pipeline.pipeline import estimate_memory_mb
    res: dict = {"baseline_mb": round(peak_rss_mb(), 1)}
    try:
        t = time.perf_counter()
        li = load_image(image)
        h, w = li.rgb.shape[:2]
        res.update(w=w, h=h, mp=round(w * h / 1e6, 1), load_s=round(time.perf_counter() - t, 2), load_peak_mb=round(peak_rss_mb(), 1))
        mask = np.zeros((h, w), np.uint8)
        mh, mw = int(h * math.sqrt(mask_frac)), int(w * math.sqrt(mask_frac))
        y0, x0 = (h - mh) // 2, (w - mw) // 2
        mask[y0:y0 + mh, x0:x0 + mw] = 255
        if model_name:
            from fixpic.cli import DEFAULT_MODELS
            from fixpic.models.manager import ModelManager
            model = ModelManager(DEFAULT_MODELS).load(model_name, threads=threads)
        else:
            model = SyntheticModel()
        t = time.perf_counter()
        out = Pipeline(model, PipelineConfig()).run(li.rgb, mask, size)
        res.update(run_s=round(time.perf_counter() - t, 2), run_peak_mb=round(peak_rss_mb(), 1), downscaled=bool(out.roi.downscaled),
                   mask_pct=round(100 * mask_frac, 1), estimate_mb=estimate_memory_mb(h, w, 0))
        t = time.perf_counter()
        sr = save_image(out_dir / "out.jpg", out.image, None, quality=90)
        res.update(save_s=round(time.perf_counter() - t, 2), save_peak_mb=round(peak_rss_mb(), 1), out_mb=round(sr.bytes_written / MB, 1))
        res["ok"] = True
    except (FixPicError, MemoryError) as e:
        res.update(ok=False, error=f"{type(e).__name__}: {e}"[:200], peak_mb=round(peak_rss_mb(), 1))
    return res


# ------------------------------------------------------------------------------------------------ ריצה ראשית
def run_one(mp: float, mask_frac: float, size: int, model: str | None, threads: int, budget_mb: float, tmp: Path) -> dict:
    w, h = dims_for_mp(mp)
    row = {"mp_target": mp, "mask_frac": mask_frac}
    need = needed_mb(w, h)
    row["needed_mb"] = need
    if need > budget_mb:
        return {**row, "ok": None, "skipped": f"ההערכה {need}MB חורגת מהתקציב {budget_mb:.0f}MB"}
    img = tmp / f"stress_{w}x{h}.jpg"
    if not img.exists():
        subprocess.run([sys.executable, __file__, "--gen", str(img), "--wh", f"{w}x{h}"], check=True, timeout=1800)
    cmd = [sys.executable, __file__, "--child", str(img), "--mask-frac", str(mask_frac), "--size", str(size), "--threads", str(threads),
           "--out-dir", str(tmp)] + (["--model", model] if model else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=3600)
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not last:
        return {**row, "ok": False, "error": f"התהליך נפל (קוד {p.returncode}): {(p.stderr or p.stdout)[-300:]}"}
    return {**row, **json.loads(last[-1])}


def to_markdown(rows: list[dict], info: dict) -> str:
    out = [f"{info['platform']}, Python {info['python']}, זיכרון פיזי {info['phys_mb']:.0f}MB, תקציב {info['budget_mb']:.0f}MB, "
           f"מודל: {info['model']}, bucket {info['size']}", "",
           "| MP | מסכה% | טעינה s | הסרה s | שמירה s | בסיס MB | שיא טעינה | שיא הסרה | שיא שמירה | הערכת Pipeline | מצב |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            out.append(f"| {r['mp_target']} | {r['mask_frac'] * 100:.0f} | - | - | - | - | - | - | - | - | דולג: {r['skipped']} |")
        elif not r.get("ok"):
            out.append(f"| {r['mp_target']} | {r['mask_frac'] * 100:.0f} | - | - | - | - | - | - | - | - | נכשל: {r.get('error', '')} |")
        else:
            out.append(f"| {r['mp']} | {r['mask_pct']:.0f} | {r['load_s']} | {r['run_s']} | {r['save_s']} | {r['baseline_mb']} | {r['load_peak_mb']} | "
                       f"{r['run_peak_mb']} | {r['save_peak_mb']} | {r['estimate_mb']} | ok{' (הוקטן)' if r['downscaled'] else ''} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mps", type=float, nargs="+", default=[12, 24, 50, 100, 150])
    ap.add_argument("--mask-fracs", type=float, nargs="+", default=[0.1])
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--model", help="שם מודל אמיתי (למשל big-lama); ברירת מחדל: מודל סינתטי")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--budget-mb", type=float, help="ברירת מחדל: 70%% מהזיכרון הפיזי")
    ap.add_argument("--tmp", type=Path, help="תיקיית עבודה לתמונות (ברירת מחדל: זמנית)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--gen", type=Path)
    ap.add_argument("--wh")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--mask-frac", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    if a.gen:
        w, h = (int(x) for x in a.wh.split("x"))
        gen_image(a.gen, w, h)
        return 0
    if a.child:
        print(json.dumps(measure_child(a.child, a.mask_frac, a.size, a.model, a.threads, a.out_dir)))
        return 0
    import tempfile
    phys = physical_memory_mb()
    budget = a.budget_mb if a.budget_mb else 0.7 * phys
    tmp_ctx = tempfile.TemporaryDirectory() if not a.tmp else None
    tmp = Path(tmp_ctx.name) if tmp_ctx else a.tmp
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for mp in a.mps:
            for frac in a.mask_fracs:
                r = run_one(mp, frac, a.size, a.model, a.threads, budget, tmp)
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()
    info = {"platform": sys.platform, "python": sys.version.split()[0], "phys_mb": phys, "budget_mb": budget,
            "model": a.model or "synthetic", "size": a.size}
    md = to_markdown(rows, info)
    print("\n" + md)
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps({"info": info, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(r.get("ok") is False for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
