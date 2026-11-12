#!/usr/bin/env python3
"""M1.5 follow-up: ONNX Runtime CPU vs OpenVINO, and 3 vs 4 threads.

Same measurement as latency_spike.py (one ROI stroke split into Pre / Infer /
Post), reusing its helpers, so the numbers are comparable with
latency_results.json. Backends measured in ONE process, interleaved by backend,
so background noise is shared:

  ort-cpu  t<N>   ONNX Runtime, CPUExecutionProvider, N intra-op threads
  ov-cpu   t<N>   OpenVINO native API, CPU, N threads, LATENCY hint, f32
  ov-gpu          OpenVINO on the Intel iGPU (only with --devices CPU GPU), f32

OpenVINO needs a static shape per ROI size, so every (size) is compiled
separately; the compile time is reported on its own because a UI that changes
the ROI size on every stroke pays it. The app plan (budget modes 256/384/512)
uses a few fixed sizes, so a compile cache keyed by size is the expected design.

Install (Python 3.14 is supported by the openvino wheel; onnxruntime-openvino is
NOT needed and has no 3.14 wheel):
  .venv\\Scripts\\python.exe -m pip install openvino

Usage (from the repo root)
  python tools\\m1\\openvino_spike.py                  sizes 256 384 512, threads 3 4 (about 5-8 min)
  python tools\\m1\\openvino_spike.py --big            adds 768 (slow)
  python tools\\m1\\openvino_spike.py --devices CPU GPU   also try the Intel iGPU
  python tools\\m1\\openvino_spike.py --dynamic        also one dynamic-shape OpenVINO compile
  python tools\\m1\\openvino_spike.py --sizes 256 --threads 4 --repeats 3   (smoke test)

Before running: plug in power, close heavy programs, do not use the computer.
Writes openvino_results.json (--out). Send that file back with the console output.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m1_check as mc  # noqa: E402  (MULTIPLE, imread, peak_mb: same padding rules as M1)
import latency_spike as ls  # noqa: E402  (make_input, timed_run, system_info)

try:
    import openvino as ov
    OV_IMPORT_ERROR = None
except Exception as exc:  # not installed or no wheel for this Python
    ov = None
    OV_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

DEFAULT_SIZES = [256, 384, 512]
CHECK_TOLERANCE = 3.0  # max abs difference (0-255 scale) before we flag a mismatch


# ------------------------------------------------------------ runners (same .run API as ORT)
class OrtRunner:
    def __init__(self, sess):
        self.sess = sess

    def run(self, names, feed):
        return self.sess.run(names, feed)


class OvRunner:
    def __init__(self, compiled):
        self.req = compiled.create_infer_request()
        self.p_rgb = compiled.input("rgb")
        self.p_mask = compiled.input("mask")
        self.p_out = compiled.output(0)

    def run(self, names, feed):
        res = self.req.infer({self.p_rgb: feed["rgb"], self.p_mask: feed["mask"]})
        return [np.array(res[self.p_out])]


def padded(side):
    return side + (-side) % mc.MULTIPLE


def ort_session(threads, model):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])


def ov_compile(core, model_path, device, threads, side, dynamic=False):
    """Read, reshape to a static (or dynamic) shape, compile. Returns (runner, compile_seconds)."""
    t = time.perf_counter()
    m = core.read_model(str(model_path))
    if dynamic:
        m.reshape({"rgb": [1, 3, -1, -1], "mask": [1, 1, -1, -1]})
    else:
        p = padded(side)
        m.reshape({"rgb": [1, 3, p, p], "mask": [1, 1, p, p]})
    cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": ov.Type.f32}
    if device == "CPU":
        cfg["INFERENCE_NUM_THREADS"] = threads
        cfg["NUM_STREAMS"] = 1
    compiled = core.compile_model(m, device, cfg)
    return OvRunner(compiled), time.perf_counter() - t


# ------------------------------------------------------------ correctness check
def raw_output(runner, img, mbin):
    """Inpainted float image (0-1) for the padded input, cropped back to the ROI."""
    h, w = img.shape[:2]
    ph, pw = (-h) % mc.MULTIPLE, (-w) % mc.MULTIPLE
    im = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    mk = cv2.copyMakeBorder(mbin, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    m = mk.astype(np.float32)[None, None]
    out = runner.run(["inpainted"], {"rgb": x, "mask": m})[0]
    return out[0].transpose(1, 2, 0)[:h, :w]


def compare(ref, got, mbin):
    sel = mbin.astype(bool)
    d = np.abs(ref - got)[sel] * 255.0
    return {"max_abs_255": round(float(d.max()), 3), "mean_abs_255": round(float(d.mean()), 4)}


# ------------------------------------------------------------ one backend over all sizes
def bench(label, make_runner, sizes, src, repeats, max_run, ref):
    print(f"\n=== {label} ===", flush=True)
    rows = []
    for side in sizes:
        row = {"backend": label, "side": side, "megapixels": round(side * side / 1e6, 3)}
        try:
            runner, compile_s = make_runner(side)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rows.append(row)
            print(f"  {side:>4}x{side:<4} FAILED to compile: {row['error']}", flush=True)
            continue
        img, mk = ls.make_input(src, side)
        try:
            ls.timed_run(runner, *ls.make_input(src, 64))  # init at a tiny shape (ORT/dynamic only matters)
        except Exception:
            pass  # static OpenVINO model rejects the 64 shape: expected
        try:
            first = ls.timed_run(runner, img, mk)
            runs = [first]
            capped = sum(first) > max_run
            for _ in range(0 if capped else repeats):
                runs.append(ls.timed_run(runner, img, mk))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            rows.append(row)
            print(f"  {side:>4}x{side:<4} FAILED to run: {row['error']}", flush=True)
            continue
        rest = runs[1:] or runs
        tot = [sum(r) for r in rest]
        med = statistics.median(tot)
        row.update({
            "compile_s": round(compile_s, 3), "first_s": round(sum(first), 3), "n": len(rest),
            "total_median_s": round(med, 3), "total_min_s": round(min(tot), 3),
            "total_max_s": round(max(tot), 3),
            "infer_median_s": round(statistics.median(r[1] for r in rest), 3),
            "spread_pct": round((max(tot) - min(tot)) / med * 100) if med > 0 else 0,
            "peak_mb": mc.peak_mb(), "capped": capped,
        })
        if ref is not None and side == ref["side"]:
            row["vs_ort"] = compare(ref["out"], raw_output(runner, img, mk), mk)
        rows.append(row)
        extra = ""
        if "vs_ort" in row:
            v = row["vs_ort"]
            flag = "  <-- MISMATCH" if v["max_abs_255"] > CHECK_TOLERANCE else ""
            extra = f" | vs ORT max {v['max_abs_255']} mean {v['mean_abs_255']} (0-255){flag}"
        print(f"  {side:>4}x{side:<4} compile {compile_s:5.2f}s | first {row['first_s']:6.2f}s | "
              f"median {med:6.2f}s (min {row['total_min_s']:.2f}, max {row['total_max_s']:.2f}, "
              f"spread {row['spread_pct']}%) | peak {row['peak_mb']}MB{extra}"
              + ("  [capped]" if capped else ""), flush=True)
        del runner
    return rows


def print_summary(all_rows, sizes):
    labels = list(dict.fromkeys(r["backend"] for r in all_rows))
    print("\n==== summary: median total seconds per stroke (Pre+Infer+Post+composite) ====")
    print("backend".ljust(18) + " | " + " | ".join(f"{s}x{s}".rjust(8) for s in sizes) + " | compile (s)")
    for lab in labels:
        cells, comp = [], []
        for s in sizes:
            r = next((x for x in all_rows if x["backend"] == lab and x["side"] == s), None)
            cells.append(("fail" if r and "error" in r else f"{r['total_median_s']:.2f}" if r else "-").rjust(8))
            if r and "compile_s" in r and r["compile_s"] > 0.05:
                comp.append(f"{r['compile_s']:.1f}")
        print(lab.ljust(18) + " | " + " | ".join(cells) + " | " + ("/".join(comp) or "-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/big-lama/inpainter.onnx")
    ap.add_argument("--image", default="samples/text.png")
    ap.add_argument("--sizes", type=int, nargs="+", help="square ROI sides (default 256 384 512)")
    ap.add_argument("--threads", type=int, nargs="+", default=[3, 4], help="thread counts (default 3 4)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-run", type=float, default=40.0, help="skip repeats if the first run exceeds this many seconds")
    ap.add_argument("--devices", nargs="+", default=["CPU"], choices=["CPU", "GPU"],
                    help="OpenVINO devices (default CPU; GPU = Intel iGPU, only if it is listed)")
    ap.add_argument("--big", action="store_true", help="also measure 768x768 (slow)")
    ap.add_argument("--dynamic", action="store_true", help="also one dynamic-shape OpenVINO CPU compile")
    ap.add_argument("--no-ort", action="store_true", help="skip the ONNX Runtime baseline")
    ap.add_argument("--out", default="openvino_results.json")
    a = ap.parse_args()

    if ov is None:
        print(f"openvino is not importable ({OV_IMPORT_ERROR}).\n"
              "Install it with:  .venv\\Scripts\\python.exe -m pip install openvino", file=sys.stderr)
        return 2
    if not Path(a.model).is_file() or not Path(a.image).is_file():
        print(f"model or image not found: {a.model} / {a.image}", file=sys.stderr)
        return 2
    with open(a.model, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            print("model is a Git LFS pointer, not the real file", file=sys.stderr)
            return 2

    sizes = list(a.sizes or DEFAULT_SIZES)
    if a.big and 768 not in sizes:
        sizes.append(768)
    core = ov.Core()
    devices = {}
    for d in core.available_devices:
        try:
            devices[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            devices[d] = "?"
    src = cv2.cvtColor(mc.imread(a.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    info = ls.system_info()
    info["openvino"] = ov.__version__
    info["ov_devices"] = devices
    print("system:", json.dumps(info, ensure_ascii=False))
    if info.get("on_battery"):
        print("WARNING: running on battery. Plug in the charger and run again.")
    if (info.get("load_before_pct") or 0) > 15:
        print(f"WARNING: CPU load before start is {info['load_before_pct']}%. Close other programs.")
    want = [d for d in a.devices if d in devices]
    for d in a.devices:
        if d not in devices:
            print(f"note: OpenVINO device {d} is not available here (found: {list(devices)}); skipped")

    # Reference output (ORT, most threads) on the first size, for the correctness check.
    ref_threads = max(a.threads)
    ref_sess = ort_session(ref_threads, a.model)
    img0, mk0 = ls.make_input(src, sizes[0])
    ref = {"side": sizes[0], "out": raw_output(OrtRunner(ref_sess), img0, mk0)}

    plan = []
    if not a.no_ort:
        plan += [("ort-cpu", t) for t in a.threads]
    for d in want:
        plan += [(f"ov-{d.lower()}", t) for t in (a.threads if d == "CPU" else [None])]
    if a.dynamic and "CPU" in want:
        plan.append(("ov-cpu-dynamic", max(a.threads)))
    print(f"plan: sizes {sizes}, backends {[p[0] + (f' t{p[1]}' if p[1] else '') for p in plan]}, "
          f"{a.repeats} repeats + 1 first run each. Do not use the computer meanwhile.", flush=True)

    all_rows = []
    for kind, t in plan:
        if kind == "ort-cpu":
            sess = ref_sess if t == ref_threads else ort_session(t, a.model)
            all_rows += bench(f"ort-cpu t{t}", lambda s, sess=sess: (OrtRunner(sess), 0.0),
                              sizes, src, a.repeats, a.max_run, ref)
        elif kind == "ov-cpu-dynamic":
            holder = {}

            def make(s, t=t, holder=holder):
                if "r" not in holder:
                    holder["r"] = ov_compile(core, a.model, "CPU", t, s, dynamic=True)
                    return holder["r"]
                return holder["r"][0], 0.0
            all_rows += bench(f"ov-cpu t{t} dynamic", make, sizes, src, a.repeats, a.max_run, ref)
        else:
            dev = "CPU" if kind == "ov-cpu" else "GPU"
            lab = f"ov-cpu t{t}" if dev == "CPU" else "ov-gpu"
            all_rows += bench(lab, lambda s, dev=dev, t=t: ov_compile(core, a.model, dev, t, s),
                              sizes, src, a.repeats, a.max_run, ref)

    print_summary(all_rows, sizes)
    out = {"meta": info, "settings": {"sizes": sizes, "threads": a.threads, "repeats": a.repeats,
                                       "devices": want, "mask_fraction": ls.MASK_FRACTION,
                                       "image": str(a.image), "check_tolerance_255": CHECK_TOLERANCE},
           "rows": all_rows, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
