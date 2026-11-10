#!/usr/bin/env python3
"""M1 check for models/big-lama/inpainter.onnx (ONNX Runtime, CPU only).

Sub-commands
  verify  size + SHA256 against the LFS pointer values, detects an LFS pointer file
  probe   loads the model and checks input/output names, types and shapes
  run     end-to-end inpainting on sample pairs (or 3 synthetic cases) with automatic
          checks, cold/warm timing and peak memory; writes PNGs and m1_results.json
  all     verify + probe + run

Pre/post follows LaMa's own predict, as described in models/big-lama/README.md:
RGB / 255, mask = (grey > 0), symmetric padding (right/bottom) to a multiple of 8,
crop the padding, 8-bit by clip(x * 255) and truncation.

Samples: a directory holding  NAME.(png|jpg|jpeg|bmp|webp)  and  NAME_mask.(same set).
Mask: any pixel above 0 is removed.
Paths with Hebrew or other Unicode characters are supported (np.fromfile + imdecode).
"""
import argparse
import hashlib
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

# Values from the Git LFS pointer of commit 8cccc36 (docs/.../DEVELOPMENT_LOG.md).
EXPECTED_SHA256 = "ef85edd865769c03c753b5faa066be713a46a9c80c8f77954c67c0bb3a4ba2ff"
EXPECTED_SIZE = 208851664
MULTIPLE = 8
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- helpers
def imread(path, flags):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


def peak_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows
        return round((peak or mi.rss) / 1e6)
    except ImportError:
        pass
    try:
        import resource  # Linux / macOS
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except ImportError:
        return None


def ort_version_ok():
    parts = ort.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (1, 30)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self):
        self.fails = 0

    def check(self, ok, text):
        print(("  PASS  " if ok else "  FAIL  ") + text)
        if not ok:
            self.fails += 1
        return ok


# ---------------------------------------------------------------- verify
def cmd_verify(a, rep):
    print("== verify ==")
    p = Path(a.model)
    if not p.is_file():
        return rep.check(False, f"file not found: {p}")
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        return rep.check(False, "file is a Git LFS POINTER, not the model. Run: git lfs install && git lfs pull")
    rep.check(size == a.size, f"size {size} (expected {a.size})")
    digest = sha256_of(p)
    rep.check(digest == a.sha256, f"sha256 {digest} (expected {a.sha256})")
    return True


# ---------------------------------------------------------------- session / probe
def load_session(a):
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(a.model), so, providers=["CPUExecutionProvider"])
    return sess, time.perf_counter() - t0


def cmd_probe(a, rep):
    print("== probe ==")
    rep.check(ort_version_ok(), f"onnxruntime {ort.__version__} (README requires 1.30 or later)")
    sess, dt = load_session(a)
    print(f"  loaded in {dt:.2f}s | providers: {sess.get_providers()}")
    ins = {i.name: i for i in sess.get_inputs()}
    outs = {o.name: o for o in sess.get_outputs()}
    for i in ins.values():
        print(f"  input  {i.name!r} {i.type} {i.shape}")
    for o in outs.values():
        print(f"  output {o.name!r} {o.type} {o.shape}")
    rep.check(set(ins) == {"rgb", "mask"}, f"input names == rgb, mask (got {sorted(ins)})")
    rep.check(list(outs) == ["inpainted"], f"output names == inpainted (got {list(outs)})")
    if "rgb" in ins:
        s = ins["rgb"].shape
        rep.check(ins["rgb"].type == "tensor(float)" and len(s) == 4 and s[1] == 3, f"rgb is float [1,3,H,W]: {ins['rgb'].type} {s}")
        rep.check(not isinstance(s[2], int) and not isinstance(s[3], int), "H and W are dynamic")
    if "mask" in ins:
        s = ins["mask"].shape
        rep.check(ins["mask"].type == "tensor(float)" and len(s) == 4 and s[1] == 1, f"mask is float [1,1,H,W]: {ins['mask'].type} {s}")
    return sess


# ---------------------------------------------------------------- samples
def synthetic_cases():
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(0)
    base = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], -1).astype(np.int16)
    base = np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
    for k in range(12):  # a few shapes so there is some structure to continue
        cv2.rectangle(base, (int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80))),
                      (int(rng.integers(80, w)), int(rng.integers(80, h))), tuple(int(c) for c in rng.integers(0, 255, 3)), 3)
    blob = np.zeros((h, w), np.uint8)
    cv2.circle(blob, (w // 2, h // 2), 60, 255, -1)
    scratch = np.zeros((h, w), np.uint8)
    cv2.line(scratch, (40, 60), (w - 60, h - 80), 255, 5)
    text = np.zeros((h, w), np.uint8)
    cv2.putText(text, "TEXT", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 255, 12)
    return [("synthetic_blob", base, blob), ("synthetic_scratch", base, scratch), ("synthetic_text", base, text)]


def discover_cases(d):
    d = Path(d)
    cases = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMG_EXT or p.stem.endswith("_mask"):
            continue
        mp = next((q for q in d.iterdir() if q.stem == p.stem + "_mask" and q.suffix.lower() in IMG_EXT), None)
        if mp is None:
            print(f"  skip {p.name}: no {p.stem}_mask.* next to it")
            continue
        cases.append((p.stem, cv2.cvtColor(imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                      imread(mp, cv2.IMREAD_GRAYSCALE)))
    return cases


# ---------------------------------------------------------------- inference
def infer(sess, img, mbin):
    """LaMa predict as in the model README. Returns float output (H,W,3) in 0..1, cropped."""
    h, w = img.shape[:2]
    ph, pw = (-h) % MULTIPLE, (-w) % MULTIPLE
    # BORDER_REFLECT == symmetric padding (edge pixel repeated), as in LaMa predict.
    im = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    mk = cv2.copyMakeBorder(mbin, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    x = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    m = mk.astype(np.float32)[None, None]
    out = sess.run(["inpainted"], {"rgb": x, "mask": m})[0]
    return out[0].transpose(1, 2, 0)[:h, :w]


def run_case(sess, name, img, mask, a, rep, outdir):
    print(f"-- {name}: {img.shape[1]}x{img.shape[0]}")
    if mask.shape != img.shape[:2]:
        return rep.check(False, f"{name}: mask size {mask.shape[::-1]} differs from image {img.shape[1::-1]}") and None
    mbin = (mask > 0).astype(np.uint8)
    if not mbin.any():
        return rep.check(False, f"{name}: empty mask") and None

    times = []
    for _ in range(1 + a.warm_runs):
        t = time.perf_counter()
        out = infer(sess, img, mbin)
        times.append(time.perf_counter() - t)
    cold, warm = times[0], (statistics.median(times[1:]) if len(times) > 1 else None)

    rep.check(out.shape == (*img.shape[:2], 3), f"output shape {out.shape}")
    rep.check(bool(np.isfinite(out).all()), "no NaN / Inf")
    lo, hi = float(out.min()), float(out.max())
    rep.check(-0.05 <= lo and hi <= 1.05, f"output range {lo:.3f} .. {hi:.3f} (expected about 0..1)")

    raw8 = (out * 255.0).clip(0, 255).astype(np.uint8)  # LaMa's conversion
    outside = mbin == 0
    diff_out = np.abs(raw8[outside].astype(int) - img[outside].astype(int))
    max_diff = int(diff_out.max()) if diff_out.size else 0
    rep.check(max_diff <= 1, f"raw model output outside mask differs from the original by at most {max_diff} level(s) (truncation allows 1)")
    final = np.where(mbin[..., None] == 1, raw8, img)
    rep.check(np.array_equal(final[outside], img[outside]), "after compositing, pixels outside the mask are identical to the original")
    changed = int((raw8[mbin == 1] != img[mbin == 1]).any(axis=-1).sum())
    rep.check(changed > 0.5 * int(mbin.sum()), f"inside-mask pixels changed: {changed}/{int(mbin.sum())}")

    mem = peak_mb()
    print(f"  cold {cold:.2f}s | warm {('%.2fs' % warm) if warm is not None else 'n/a'} | peak memory {mem} MB")

    outdir.mkdir(parents=True, exist_ok=True)
    imwrite(outdir / f"{name}_result.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    overlay = img.copy()
    overlay[mbin == 1] = (0.5 * overlay[mbin == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    side = np.hstack([img, overlay, final])
    imwrite(outdir / f"{name}_compare.png", cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    return {"name": name, "width": img.shape[1], "height": img.shape[0], "mask_pixels": int(mbin.sum()),
            "cold_s": round(cold, 3), "warm_s": None if warm is None else round(warm, 3),
            "peak_mb": mem, "max_diff_outside_raw": max_diff}


def cmd_run(a, rep, sess=None, load_s=None):
    print("== run ==")
    if sess is None:
        sess, load_s = load_session(a)
    if a.dir:
        cases = discover_cases(a.dir)
        if not cases:
            return rep.check(False, f"no image + NAME_mask pairs found in {a.dir}")
    else:
        print("  no --dir given: using 3 synthetic cases (plumbing only, not a visual-quality test)")
        cases = synthetic_cases()
    results = [r for r in (run_case(sess, n, i, m, a, rep, Path(a.out)) for n, i, m in cases) if r]
    meta = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
            "onnxruntime": ort.__version__, "threads_setting": a.threads or "default", "model_load_s": round(load_s, 3)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "m1_results.json").write_text(json.dumps({"meta": meta, "cases": results}, indent=2), encoding="utf-8")
    print(f"  wrote {Path(a.out) / 'm1_results.json'}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify", "probe", "run", "all"])
    ap.add_argument("--model", default="models/big-lama/inpainter.onnx")
    ap.add_argument("--dir", help="folder with NAME.* and NAME_mask.* pairs")
    ap.add_argument("--out", default="m1_out")
    ap.add_argument("--threads", type=int, default=0, help="0 = ONNX Runtime default")
    ap.add_argument("--warm-runs", type=int, default=2)
    ap.add_argument("--sha256", default=EXPECTED_SHA256)
    ap.add_argument("--size", type=int, default=EXPECTED_SIZE)
    a = ap.parse_args()

    rep = Report()
    if a.cmd in ("verify", "all"):
        if not cmd_verify(a, rep) or rep.fails:
            if a.cmd == "all":
                print("\nverify failed - stopping (probe/run need the real model).")
                sys.exit(1)
    if a.cmd == "probe":
        cmd_probe(a, rep)
    if a.cmd == "all":
        cmd_probe(a, rep)
    if a.cmd in ("run", "all"):
        cmd_run(a, rep)
    print(f"\n{'ALL CHECKS PASSED' if rep.fails == 0 else str(rep.fails) + ' CHECK(S) FAILED'}")
    sys.exit(1 if rep.fails else 0)


if __name__ == "__main__":
    main()
