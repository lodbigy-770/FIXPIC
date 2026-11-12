#!/usr/bin/env python3
"""M1.5 follow-up (session A): GIL during inference, and start-up (import) time.

Part 1 - start-up: fresh `python -c` processes time `import X` (median of 3) for
numpy, cv2, onnxruntime, openvino, PySide6.QtWidgets (skipped if not installed)
and the engine (`fixpic.pipeline`, with PYTHONPATH=src) next to bare interpreter start.

Part 2 - GIL: a worker thread runs real inference (OpenVINO and ORT, static
shape, N threads) while the MAIN thread, standing in for the UI, does either
  sleep  - sleeps 2 ms in a loop and records how late it wakes (UI event loop idle)
  busy   - pure-Python work in 20 ms slices and counts slices per second
Baseline (no inference) vs during inference. If inference held the GIL, the main
thread would stall for whole seconds (max wake delay ~ inference time).

  python tools\\m1\\gil_startup_spike.py                 about 2-3 minutes
  python tools\\m1\\gil_startup_spike.py --size 384 --threads 3 --runs 3

Writes gil_startup.json (--out). Do not use the computer meanwhile.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))


def import_time(stmt, repeats=3):
    code = f"import time;t=time.perf_counter();{stmt};print(time.perf_counter()-t)"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    vals = []
    for _ in range(repeats):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        if r.returncode:
            return None
        vals.append(float(r.stdout.strip().splitlines()[-1]))
    return round(statistics.median(vals), 3)


def startup():
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-c", "pass"], capture_output=True)
    out = {"interpreter_start": round(time.perf_counter() - t0, 3)}
    for label, stmt in [("numpy", "import numpy"), ("cv2", "import cv2"),
                        ("onnxruntime", "import onnxruntime"), ("openvino", "import openvino"),
                        ("PySide6.QtWidgets", "import PySide6.QtWidgets"),
                        ("fixpic.pipeline", "import fixpic.pipeline")]:
        out[label] = import_time(stmt)
    return out


def main_thread_probe(mode, stop_evt, seconds):
    """Runs in the main thread until stop_evt is set (or `seconds` pass). Returns stats."""
    end = time.perf_counter() + seconds
    if mode == "sleep":
        delays = []
        while not stop_evt.is_set() and time.perf_counter() < end:
            t = time.perf_counter()
            time.sleep(0.002)
            delays.append(time.perf_counter() - t - 0.002)
        delays.sort()
        n = len(delays)
        return {"samples": n, "median_ms": round(delays[n // 2] * 1e3, 2),
                "p99_ms": round(delays[int(n * 0.99)] * 1e3, 2), "max_ms": round(delays[-1] * 1e3, 2)}
    slices, t_start = 0, time.perf_counter()
    while not stop_evt.is_set() and time.perf_counter() < end:
        t = time.perf_counter()
        while time.perf_counter() - t < 0.02:
            sum(range(500))
        slices += 1
    return {"slices_per_s": round(slices / (time.perf_counter() - t_start), 1)}


def probe(mode, worker=None, idle_s=2.0):
    """Baseline when worker is None; otherwise probe while the worker thread runs."""
    stop = threading.Event()
    if worker is None:
        return main_thread_probe(mode, stop, idle_s)
    th = threading.Thread(target=lambda: (worker(), stop.set()))
    th.start()
    res = main_thread_probe(mode, stop, 600)
    th.join()
    return res


def gil_part(a):
    import numpy as np
    import openvino_spike as osp
    import m1_check as mc
    p = a.size + (-a.size) % mc.MULTIPLE
    rng = np.random.default_rng(0)
    feed = {"rgb": rng.random((1, 3, p, p), dtype=np.float32),
            "mask": (rng.random((1, 1, p, p)) > 0.9).astype(np.float32)}
    runners = {}
    if osp.ov is not None:
        core = osp.ov.Core()
        runners[f"ov-cpu t{a.threads}"] = osp.ov_compile(core, a.model, "CPU", a.threads, a.size)[0]
    runners[f"ort-cpu t{a.threads}"] = osp.OrtRunner(osp.ort_session(a.threads, a.model))
    res = {}
    for name, r in runners.items():
        r.run(["inpainted"], feed)  # warm
        t = time.perf_counter()
        r.run(["inpainted"], feed)
        one = time.perf_counter() - t

        def worker(r=r):
            for _ in range(a.runs):
                r.run(["inpainted"], feed)

        row = {"infer_s": round(one, 3)}
        for mode in ("sleep", "busy"):
            row[mode] = {"baseline": probe(mode), "during": probe(mode, worker)}
        res[name] = row
        print(f"\n{name}: one inference {one:.2f}s")
        s, b = row["sleep"], row["busy"]
        print(f"  main thread sleep(2ms): baseline p99 {s['baseline']['p99_ms']} ms, max {s['baseline']['max_ms']} ms | "
              f"during p99 {s['during']['p99_ms']} ms, max {s['during']['max_ms']} ms")
        print(f"  main thread busy work : baseline {b['baseline']['slices_per_s']}/s | during {b['during']['slices_per_s']}/s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/big-lama/inpainter.onnx")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3, help="inferences per probe while the worker runs")
    ap.add_argument("--out", default="gil_startup.json")
    ap.add_argument("--skip-startup", action="store_true")
    a = ap.parse_args()
    out = {"python": sys.version.split()[0]}
    if not a.skip_startup:
        print("== start-up: import time in fresh processes (seconds, median of 3)")
        out["startup"] = startup()
        for k, v in out["startup"].items():
            print(f"  {k:<20} {v if v is not None else 'not installed / failed'}")
    print("\n== GIL: main thread while a worker thread runs inference")
    out["gil"] = gil_part(a)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
