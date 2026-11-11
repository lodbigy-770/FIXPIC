#!/usr/bin/env python3
"""M1.5 Latency Spike: how long does one brush stroke take on this CPU?

Measures Big LaMa (ONNX Runtime, CPU) for square ROI sizes and thread counts,
and splits every run into Pre / Infer / Post so we know what is worth optimizing.

Usage (from the repo root)
  python tools\\m1\\latency_spike.py --quick        about 5 minutes
  python tools\\m1\\latency_spike.py                about 15 minutes (default sizes)
  python tools\\m1\\latency_spike.py --big          adds 1024x1024 (slow)
  python tools\\m1\\latency_spike.py --sizes 256 512 --threads 1 4 --repeats 3

Before running: plug the laptop into power, close heavy programs (browser, IDE
indexing, antivirus scans), and do not use the computer while it runs. The tool
prints warnings for load and battery, because timings on this machine have been
noisy (the same image took 6.6 s and 9.5 s).

Writes latency_results.json (--out). Send that file back.
Timing content does not depend on the picture, only on shapes, so the sample
image (default samples/text.png) is just cropped or resized to each ROI size.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_check as mc  # reuse imread / peak_mb / MULTIPLE (same padding rules as M1)

DEFAULT_SIZES = [128, 256, 384, 512, 768]
QUICK_SIZES = [256, 384, 512]
MASK_FRACTION = 0.30  # side of the centered square mask relative to the ROI


# ------------------------------------------------------------ system info
def cpu_name():
    try:  # Windows: the readable name, e.g. "Intel(R) Core(TM) i5-..."
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "unknown"


def system_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "cpu": cpu_name(),
            "logical_cores": os.cpu_count(), "physical_cores": None,
            "cpu_mhz_max": None, "ram_gb": None, "on_battery": None, "load_before_pct": None}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        f = psutil.cpu_freq()
        info["cpu_mhz_max"] = round(f.max) if f and f.max else None
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        b = psutil.sensors_battery()
        info["on_battery"] = None if b is None else (not b.power_plugged)
        info["load_before_pct"] = psutil.cpu_percent(interval=2.0)
    except Exception:
        pass
    return info


# ------------------------------------------------------------ inputs
def make_input(src_rgb, side):
    """Square RGB of the given side: crop when the source is big enough, else resize to cover."""
    h, w = src_rgb.shape[:2]
    if min(h, w) >= side:
        y, x = (h - side) // 2, (w - side) // 2
        img = src_rgb[y:y + side, x:x + side].copy()
    else:
        s = side / min(h, w)
        big = cv2.resize(src_rgb, (max(side, int(round(w * s))), max(side, int(round(h * s)))),
                         interpolation=cv2.INTER_LINEAR)
        y, x = (big.shape[0] - side) // 2, (big.shape[1] - side) // 2
        img = big[y:y + side, x:x + side].copy()
    mask = np.zeros((side, side), np.uint8)
    m = max(2, int(side * MASK_FRACTION))
    o = (side - m) // 2
    mask[o:o + m, o:o + m] = 1
    return img, mask


# ------------------------------------------------------------ one timed run
def timed_run(sess, img, mbin):
    """Same steps as m1_check.infer plus the composite, timed in three parts (seconds)."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    ph, pw = (-h) % mc.MULTIPLE, (-w) % mc.MULTIPLE
    im = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    mk = cv2.copyMakeBorder(mbin, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    m = mk.astype(np.float32)[None, None]
    t1 = time.perf_counter()
    out = sess.run(["inpainted"], {"rgb": x, "mask": m})[0]
    t2 = time.perf_counter()
    res = out[0].transpose(1, 2, 0)[:h, :w]
    res8 = np.clip(res * 255.0, 0, 255).astype(np.uint8)   # truncate, as in the README
    sel = mbin.astype(bool)[:, :, None]
    final = np.where(sel, res8, img)                       # composite: orig*(1-m)+result*m
    t3 = time.perf_counter()
    assert final.shape == img.shape
    return t1 - t0, t2 - t1, t3 - t2


def new_session(threads):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    t = time.perf_counter()
    s = ort.InferenceSession(str(ARGS.model), so, providers=["CPUExecutionProvider"])
    return s, time.perf_counter() - t


def med(v):
    return statistics.median(v)


def bench_config(threads, sizes, src, repeats, max_run):
    sess, load_s = new_session(threads)
    label = threads if threads else "default"
    print(f"\n=== threads={label} (model load {load_s:.2f}s) ===", flush=True)
    # Run a tiny input once so one-time session init is not charged to the first size.
    img, mk = make_input(src, 64)
    timed_run(sess, img, mk)
    rows = []
    for side in sizes:
        img, mk = make_input(src, side)
        first = timed_run(sess, img, mk)  # first call at a NEW shape (what a UI stroke often is)
        runs = [first]
        budget_hit = sum(first) > max_run
        for _ in range(0 if budget_hit else repeats):
            runs.append(timed_run(sess, img, mk))
        rest = runs[1:] or runs
        pre, inf, post = ([r[i] for r in rest] for i in range(3))
        tot = [sum(r) for r in rest]
        row = {"threads": label, "side": side, "megapixels": round(side * side / 1e6, 3),
               "first_s": round(sum(first), 3), "n": len(rest),
               "total_median_s": round(med(tot), 3), "total_min_s": round(min(tot), 3),
               "total_max_s": round(max(tot), 3),
               "pre_median_ms": round(1000 * med(pre), 1),
               "infer_median_s": round(med(inf), 3),
               "post_median_ms": round(1000 * med(post), 1),
               "peak_mb": mc.peak_mb(), "capped": budget_hit}
        row["spread_pct"] = round((max(tot) - min(tot)) / med(tot) * 100) if med(tot) > 0 else 0
        rows.append(row)
        print(f"  {side:>4}x{side:<4} first {row['first_s']:6.2f}s | median {row['total_median_s']:6.2f}s "
              f"(min {row['total_min_s']:.2f}, max {row['total_max_s']:.2f}, spread {row['spread_pct']}%) | "
              f"pre {row['pre_median_ms']:.0f}ms infer {row['infer_median_s']:.2f}s post {row['post_median_ms']:.0f}ms | "
              f"peak {row['peak_mb']}MB" + ("  [capped: slow, fewer repeats]" if budget_hit else ""),
              flush=True)
    del sess
    return {"threads": label, "model_load_s": round(load_s, 3), "rows": rows}


def resize_costs(src):
    """Cost of the 'big ROI' strategy: downscale 1536->512, later upscale back (ms)."""
    big = cv2.resize(src, (1536, 1536), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(big, (512, 512), interpolation=cv2.INTER_AREA)

    def t(fn, n=5):
        v = []
        for _ in range(n):
            a = time.perf_counter()
            fn()
            v.append(time.perf_counter() - a)
        return round(1000 * med(v), 1)
    return {"downscale_1536_to_512_area_ms": t(lambda: cv2.resize(big, (512, 512), interpolation=cv2.INTER_AREA)),
            "upscale_512_to_1536_lanczos_ms": t(lambda: cv2.resize(small, (1536, 1536), interpolation=cv2.INTER_LANCZOS4))}


def print_table(results):
    print("\n==== summary: median total seconds per stroke (Pre+Infer+Post+composite) ====")
    sizes = sorted({r["side"] for c in results for r in c["rows"]})
    print("threads | " + " | ".join(f"{s}x{s}" for s in sizes))
    for c in results:
        by = {r["side"]: r for r in c["rows"]}
        cells = [f"{by[s]['total_median_s']:.2f}" if s in by else "-" for s in sizes]
        print(f"{str(c['threads']):>7} | " + " | ".join(cells))


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/big-lama/inpainter.onnx")
    ap.add_argument("--image", default="samples/text.png", help="any image; only its shape matters for timing")
    ap.add_argument("--sizes", type=int, nargs="+", help="square ROI sides (default 128 256 384 512 768)")
    ap.add_argument("--threads", type=int, nargs="+", help="thread counts (default 1 2 <logical cores>)")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per size after the first (default 5)")
    ap.add_argument("--max-run", type=float, default=40.0,
                    help="if the first run of a size takes longer than this (s), skip its repeats")
    ap.add_argument("--quick", action="store_true", help="sizes 256 384 512, threads 1 and all, 3 repeats")
    ap.add_argument("--big", action="store_true", help="also measure 1024x1024 (slow)")
    ap.add_argument("--out", default="latency_results.json")
    ARGS = a = ap.parse_args()

    cores = os.cpu_count() or 4
    sizes = a.sizes or (QUICK_SIZES if a.quick else list(DEFAULT_SIZES))
    if a.big and 1024 not in sizes:
        sizes.append(1024)
    threads = a.threads or ([1, cores] if a.quick else sorted({1, 2, cores}))
    repeats = 3 if (a.quick and a.repeats == 5) else a.repeats
    if any(s % 8 for s in sizes):
        print("note: sizes that are not multiples of 8 are padded by the same code path as the app")
    if not Path(a.model).is_file():
        print(f"model not found: {a.model}", file=sys.stderr)
        return 2
    if not Path(a.image).is_file():
        print(f"image not found: {a.image}", file=sys.stderr)
        return 2
    with open(a.model, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            print("model is a Git LFS pointer, not the real file", file=sys.stderr)
            return 2

    src = cv2.cvtColor(mc.imread(a.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    info = system_info()
    print("system:", json.dumps(info, ensure_ascii=False))
    if info.get("on_battery"):
        print("WARNING: running on battery. Plug in the charger, then run again for clean numbers.")
    if (info.get("load_before_pct") or 0) > 15:
        print(f"WARNING: CPU load before start is {info['load_before_pct']}%. Close other programs.")
    print(f"plan: sizes {sizes}, threads {threads}, {repeats} repeats + 1 first run each. "
          "Do not use the computer meanwhile.", flush=True)

    results = [bench_config(t, sizes, src, repeats, a.max_run) for t in threads]
    rc = resize_costs(src)
    print("\nresize costs (strategy for ROI larger than the model input):", rc)
    print_table(results)

    out = {"meta": info, "settings": {"sizes": sizes, "threads": threads, "repeats": repeats,
                                       "mask_fraction": MASK_FRACTION, "image": str(a.image)},
           "configs": results, "resize_costs": rc,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
