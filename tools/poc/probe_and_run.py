#!/usr/bin/env python3
"""PoC: probe an inpainting ONNX model and run a basic CPU inpainting test.

Nothing about the model is hard-coded. Input/output names, dtypes and shapes are
read from the file. Anything that cannot be read from the file (value range,
mask polarity) is an explicit CLI flag and is printed as an ASSUMPTION.

Usage:
  python probe_and_run.py probe --model models/big-lama/<file>.onnx
  python probe_and_run.py run   --model <file>.onnx [--image a.png --mask m.png]
                                [--out out.png] [--range 01|255] [--mask-polarity 1|0]
                                [--threads N] [--multiple 8]
Without --image/--mask a synthetic test image and mask are generated.
White (>127) pixels in the mask file = area to remove.

Status: plumbing validated on a dummy model only; NOT yet run on Big LaMa.
"""
import argparse
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

NP_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def load_session(path, threads):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def describe(sess):
    print("ONNX Runtime:", ort.__version__, "| providers:", sess.get_providers())
    print("INPUTS:")
    for i in sess.get_inputs():
        print(f"  name={i.name!r} type={i.type} shape={i.shape}")
    print("OUTPUTS:")
    for o in sess.get_outputs():
        print(f"  name={o.name!r} type={o.type} shape={o.shape}")
    meta = sess.get_modelmeta()
    print("METADATA:", {"producer": meta.producer_name, "version": meta.version,
                        "description": meta.description, "custom": meta.custom_metadata_map})


def is_int(x):
    return isinstance(x, int)


def pick_inputs(sess, name_img=None, name_mask=None):
    ins = sess.get_inputs()
    if name_img and name_mask:
        return {i.name: i for i in ins}[name_img], {i.name: i for i in ins}[name_mask]
    img = mask = None
    for i in ins:
        ch = i.shape[1] if len(i.shape) == 4 else None
        if "mask" in i.name.lower() and mask is None:
            mask = i
        elif ch == 3 and img is None:
            img = i
        elif ch == 1 and mask is None:
            mask = i
    if img is None or mask is None:
        sys.exit("Could not identify image/mask inputs automatically; "
                 "pass --image-input and --mask-input.")
    return img, mask


def synthetic():
    h, w = 300, 420
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], -1).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 40, 255, -1)
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image")
    ap.add_argument("--mask")
    ap.add_argument("--out", default="poc_out.png")
    ap.add_argument("--range", dest="rng", choices=["01", "255"], default="01")
    ap.add_argument("--mask-polarity", type=int, choices=[0, 1], default=1,
                    help="value the model expects for 'remove' pixels (assumption)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--multiple", type=int, default=8,
                    help="pad dynamic-shape inputs to a multiple of this (assumption)")
    ap.add_argument("--image-input")
    ap.add_argument("--mask-input")
    a = ap.parse_args()

    t0 = time.perf_counter()
    sess = load_session(a.model, a.threads)
    print(f"model loaded in {time.perf_counter() - t0:.2f}s")
    describe(sess)
    if a.cmd == "probe":
        return

    # ---- load data ----
    if a.image and a.mask:
        bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
        m = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        if bgr is None or m is None:
            sys.exit("failed to read image/mask")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        img, m = synthetic()
        print("using synthetic image + mask")
    if m.shape != img.shape[:2]:
        sys.exit("mask and image sizes differ")
    mbin = (m > 127).astype(np.uint8)
    if mbin.sum() == 0:
        sys.exit("empty mask")
    H0, W0 = img.shape[:2]

    print("ASSUMPTIONS:", f"value range={a.rng}", f"remove-value={a.mask_polarity}",
          f"dynamic-pad multiple={a.multiple}")

    iin, imask = pick_inputs(sess, a.image_input, a.mask_input)
    th, tw = iin.shape[2], iin.shape[3]
    static = is_int(th) and is_int(tw)

    # ---- resize / pad ----
    if static:
        print(f"static input {th}x{tw}: resizing whole image (PoC only)")
        im2 = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mbin, (tw, th), interpolation=cv2.INTER_NEAREST)
        pad = (0, 0)
    else:
        ph = (-H0) % a.multiple
        pw = (-W0) % a.multiple
        im2 = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        m2 = cv2.copyMakeBorder(mbin, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        pad = (ph, pw)

    scale = 1.0 if a.rng == "255" else 1.0 / 255.0
    x_img = (im2.astype(np.float32) * scale).transpose(2, 0, 1)[None]
    mval = m2 if a.mask_polarity == 1 else (1 - m2)
    x_mask = mval.astype(np.float32)[None, None]
    x_img = x_img.astype(NP_DTYPES.get(iin.type, np.float32))
    x_mask = x_mask.astype(NP_DTYPES.get(imask.type, np.float32))

    # ---- inference ----
    t1 = time.perf_counter()
    outs = sess.run(None, {iin.name: x_img, imask.name: x_mask})
    dt = time.perf_counter() - t1
    o = np.asarray(outs[0]).astype(np.float32)
    print(f"inference: {dt:.2f}s | output shape {o.shape} min={o.min():.3f} max={o.max():.3f}")

    # ---- postprocess ----
    if o.ndim == 4:
        o = o[0].transpose(1, 2, 0)
    if o.max() <= 1.5:
        print("output looks like 0..1 range -> scaling to 0..255 (heuristic, verify!)")
        o = o * 255.0
    o = np.clip(o, 0, 255).astype(np.uint8)
    if pad != (0, 0):
        o = o[:o.shape[0] - pad[0] or None, :o.shape[1] - pad[1] or None]
    if o.shape[:2] != (H0, W0):
        o = cv2.resize(o, (W0, H0), interpolation=cv2.INTER_CUBIC)

    # ---- blend + checks ----
    mm = mbin[..., None].astype(np.float32)
    final = (img * (1 - mm) + o * mm).astype(np.uint8)
    same = np.array_equal(final[mbin == 0], img[mbin == 0])
    changed = int((final[mbin == 1] != img[mbin == 1]).any(axis=-1).sum())
    print(f"outside-mask pixels identical to original: {same}")
    print(f"inside-mask pixels changed: {changed}/{int(mbin.sum())}")

    cv2.imwrite(a.out, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print("wrote", a.out)
    try:
        import resource
        print("peak RSS MB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    except ImportError:  # Windows
        pass


if __name__ == "__main__":
    main()
