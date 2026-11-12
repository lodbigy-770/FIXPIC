#!/usr/bin/env python3
"""M1.5 follow-up (session A): OpenVINO compile cost with and without CACHE_DIR.

Each phase runs in a FRESH Python process, because that is what an app restart
looks like. Phases: cold (empty cache dir) then warm (cache filled by cold).
For every shape (static 256/384/512 and one dynamic) it reports the compile
time (read_model + reshape + compile_model) and the first inference after it.

  python tools\\m1\\ov_cache_spike.py                 about 1-2 minutes
  python tools\\m1\\ov_cache_spike.py --threads 3

Writes ov_cache.json (--out). The cache dir (ov_cache_spike_dir) is deleted at
the start and at the end (--keep to leave it). Do not use the computer meanwhile.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def dir_mb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6 if Path(p).exists() else 0.0


def phase(a):
    import cv2
    import numpy as np
    import openvino as ov
    import m1_check as mc
    import latency_spike as ls

    src = cv2.cvtColor(mc.imread(a.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    t = time.perf_counter()
    core = ov.Core()
    core.set_property({"CACHE_DIR": a.cache_dir})
    import_s = time.perf_counter() - t
    rows = []
    for label, side, dynamic in [("static 256", 256, False), ("static 384", 384, False),
                                 ("static 512", 512, False), ("dynamic", 256, True)]:
        t = time.perf_counter()
        m = core.read_model(a.model)
        read_s = time.perf_counter() - t
        if dynamic:
            m.reshape({"rgb": [1, 3, -1, -1], "mask": [1, 1, -1, -1]})
        else:
            p = side + (-side) % mc.MULTIPLE
            m.reshape({"rgb": [1, 3, p, p], "mask": [1, 1, p, p]})
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": ov.Type.f32,
               "INFERENCE_NUM_THREADS": a.threads, "NUM_STREAMS": 1}
        compiled = core.compile_model(m, "CPU", cfg)
        compile_s = time.perf_counter() - t
        req = compiled.create_infer_request()
        pr, pm, po = compiled.input("rgb"), compiled.input("mask"), compiled.output(0)

        class R:
            def run(self, names, feed):
                return [np.array(req.infer({pr: feed["rgb"], pm: feed["mask"]})[po])]

        img, mk = ls.make_input(src, side)
        first = sum(ls.timed_run(R(), img, mk))
        rows.append({"label": label, "compile_s": round(compile_s, 3), "read_s": round(read_s, 3),
                     "first_s": round(first, 3)})
    print(json.dumps({"import_s": round(import_s, 3), "rows": rows}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/big-lama/inpainter.onnx")
    ap.add_argument("--image", default="samples/text.png")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--cache-dir", default="ov_cache_spike_dir")
    ap.add_argument("--out", default="ov_cache.json")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--phase", choices=["run"], help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.phase:
        return phase(a)

    shutil.rmtree(a.cache_dir, ignore_errors=True)
    results = {}
    for name in ("cold", "warm"):
        print(f"== {name} (fresh process) ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", "run", "--model", a.model,
               "--image", a.image, "--threads", str(a.threads), "--cache-dir", a.cache_dir]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr)
            sys.exit(r.returncode)
        results[name] = json.loads(r.stdout.strip().splitlines()[-1])
        results[name]["cache_mb_after"] = round(dir_mb(a.cache_dir), 1)
    print("\nshape        | cold compile | warm compile | cold first | warm first")
    for c, w in zip(results["cold"]["rows"], results["warm"]["rows"]):
        print(f"{c['label']:<12} | {c['compile_s']:>10.2f}s | {w['compile_s']:>10.2f}s | "
              f"{c['first_s']:>8.2f}s | {w['first_s']:>8.2f}s")
    print(f"cache dir size after cold: {results['cold']['cache_mb_after']} MB; "
          f"after warm: {results['warm']['cache_mb_after']} MB")
    Path(a.out).write_text(json.dumps({"threads": a.threads, **results}, indent=1), encoding="utf-8")
    print("wrote", a.out)
    if not a.keep:
        shutil.rmtree(a.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
