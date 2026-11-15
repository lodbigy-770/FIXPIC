#!/usr/bin/env python3
"""AUTO-SELECT.md §7 שלב 5: Spike מודל "לחיצה-חכמה" (Promptable Segmentation) – שער ההחלטה לפני `ModelProvider`.

הסקריפט **אינו מכריע ואינו מוריד מודלים**. הוא מודד אצלך (i5-6500T) כל מועמד שתביא כזוג קבצי ONNX
(מקודד כבד + מפענח קל, משפחת SAM) ומדפיס מול קריטריוני השער: `propose` <= 0.5s, `prepare` <= 5s ברקע,
זיכרון נוסף סביר, איכות בעין, רישיון. **כל מה שהוא מדפיס הוא מדידה; כל מה שנאמר כאן על פורמט הקלט/הפלט
של משפחת SAM [לא אומת] מול מודל אמיתי** (נבדק רק מול זוג מודלים סינתטי, `tests/test_m4_model_spike.py`).
אם חוזה המפענח של מועמד שונה, הסקריפט עוצר עם רשימת שמות הקלט/פלט שלו ולא מנחש.

הרצה מהשורש (דרוש: onnxruntime, opencv-python-headless, numpy, psutil; fixpic אופציונלי להשוואה ל-GrabCut):
  python tools/m4/model_spike.py --candidate mobilesam C:\\models\\enc.onnx C:\\models\\dec.onnx
  python tools/m4/model_spike.py --candidate A enc_a.onnx dec_a.onnx --candidate B enc_b.onnx dec_b.onnx
  python tools/m4/model_spike.py --candidate A enc.onnx dec.onnx --threads 4 1 --contention

לפני ההרצה: חיבור לחשמל, לסגור תוכנות כבדות, לא לגעת במחשב בזמן הריצה (המדידות ב-M1 היו רועשות).
פלט: `<out>/model_spike_results.json` (לשלוח חזרה) ו-`<out>/<מועמד>/<מקרה>_<סוג>.png` (לבדיקה בעין:
אדום = ההצעה הטובה ביותר, קווי מתאר אחרים = חלופות, ציאן = מלבן, ירוק = לחיצה).

נקודות מדידה:
  prepare   = Pre (הקטנה ונרמול) + Encoder. מדוד על תמונה אמיתית (0.4MP) וגם על 24MP סינתטי (התוכן לא משנה את הזמן).
  propose   = Decoder + שחזור המסכה לגודל התמונה. לחיצה במרכז המלבן, ומלבן.
  --threads = מספר threads למודל. ברירת מחדל `4 1`: 4 = המודל לבד, 1 = מה שנשאר בזמן Inpaint (ADR-011 תופס 3).
  --contention = חוזר על המדידה כש-Big LaMa רץ ברקע על 3 threads (דורש `models/big-lama/inpainter.onnx` אמיתי).
אין כאן אמת-קרקע: איכות = בדיקה בעין של ה-PNG-ים, ובנוסף IoU אינפורמטיבי מול GrabCut ובין לחיצה למלבן.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]

SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
SAM_STD = np.array([58.395, 57.12, 57.375], np.float32)

# קריטריוני השער (AUTO-SELECT §7 שלב 5). זיכרון ואיכות ורישיון: הכרעה ידנית.
GATE_PROPOSE_S = 0.5
GATE_PREPARE_S = 5.0

# מקרי ברירת מחדל = קופסאות ה-GrabCut הצמודות מ-`grabcut_samples.py` (נבדקו בעין על `samples/`).
DEFAULT_CASES = [
    {"name": "person_box", "image": "person", "box": (160, 285, 500, 668)},
    {"name": "bin_box", "image": "text", "box": (0, 285, 90, 412)},
    {"name": "sign_box", "image": "text", "box": (252, 97, 542, 158)},
    {"name": "one_sign_in_strip_box", "image": "text", "box": (258, 412, 295, 449)},
    {"name": "booth_box", "image": "text", "box": (215, 75, 580, 490)},
]


class SpikeError(Exception):
    """מועמד שאי אפשר למדוד (קובץ חסר/LFS pointer, חוזה לא מוכר)."""


# ------------------------------------------------------------------ כלים כלליים
def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SpikeError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tofile(str(path))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median_ms(times_s: list[float]) -> float:
    return round(statistics.median(times_s) * 1000, 1)


def check_model_file(path: Path) -> None:
    if not path.is_file():
        raise SpikeError(f"file not found: {path}")
    head = path.read_bytes()[:64] if path.stat().st_size < 4096 else b""
    if head.startswith(b"version https://git-lfs"):
        raise SpikeError(f"{path} is a Git LFS pointer, not the real model; download the real file")
    if path.stat().st_size == 0:
        raise SpikeError(f"{path} is empty")


def cpu_name() -> str:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


def env_info() -> dict:
    info = {"python": platform.python_version(), "os": platform.platform(), "cpu": cpu_name(),
            "onnxruntime": ort.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
            "logical_cores": os.cpu_count()}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        b = psutil.sensors_battery()
        info["on_battery"] = None if b is None else (not b.power_plugged)
        info["load_before_pct"] = psutil.cpu_percent(interval=1.0)
    except Exception:
        pass
    return info


class RssSampler:
    """שיא RSS של התהליך בחלון זמן (דגימה כל 10ms). בלי psutil: `peak_mb` = None."""

    def __init__(self, interval: float = 0.01):
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc = None
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self):
        if self._proc is not None:
            self.peak = self._proc.memory_info().rss
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float | None:
        return None if self._proc is None else round(self.peak / 1e6)


def rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6)
    except ImportError:
        return None


def license_info(model_dir: Path) -> dict:
    """קבצי רישיון/README ליד המודל. **רק רמז**: הרישיון נבדק ידנית מול מקור המודל (קוד, משקולות ונתוני אימון)."""
    files = sorted(p.name for p in model_dir.iterdir()
                   if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "README")))
    excerpt = {}
    for n in files:
        if n.upper().startswith(("LICENSE", "COPYING")):
            excerpt[n] = " ".join(model_dir.joinpath(n).read_text("utf-8", errors="replace").split())[:300]
    return {"files_next_to_model": files, "license_excerpt": excerpt, "verified": False}


# ------------------------------------------------------------------ חוזה SAM (לא אומת מול מודל אמיתי)
def preprocess(rgb: np.ndarray, size: int, norm: str, resize: str):
    """צלע ארוכה -> `size`, נרמול, ריפוד באפסים ימין/מטה. מחזיר (טנזור [1,3,size,size], scale, newh, neww)."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    newh, neww = int(h * scale + 0.5), int(w * scale + 0.5)
    interp = cv2.INTER_AREA if (resize == "area" and scale < 1) else cv2.INTER_LINEAR
    x = cv2.resize(rgb, (neww, newh), interpolation=interp).astype(np.float32)
    if norm == "sam":
        x = (x - SAM_MEAN) / SAM_STD
    elif norm == "unit":
        x = x / 255.0
    elif norm != "none":
        raise SpikeError(f"unknown --norm {norm}")
    padded = np.zeros((size, size, 3), np.float32)
    padded[:newh, :neww] = x
    return padded.transpose(2, 0, 1)[None], scale, newh, neww


def encode_prompts(points, box, scale: float):
    """points: [(x, y, positive)], box: (x0,y0,x1,y1)|None, בפיקסלי התמונה המקורית.
    SAM: חיובי=1, שלילי=0, פינות מלבן=2,3; בלי מלבן מוסיפים נקודת ריפוד (0,0) עם תווית -1."""
    coords = [(x * scale, y * scale) for x, y, _ in points]
    labels = [1.0 if pos else 0.0 for _, _, pos in points]
    if box is not None:
        x0, y0, x1, y1 = box
        coords += [(x0 * scale, y0 * scale), (x1 * scale, y1 * scale)]
        labels += [2.0, 3.0]
    else:
        coords.append((0.0, 0.0))
        labels.append(-1.0)
    return np.array(coords, np.float32)[None], np.array(labels, np.float32)[None]


def role_of(name: str) -> str | None:
    n = name.lower()
    if "has_mask" in n:
        return "has_mask"
    if "mask" in n:
        return "mask_input"
    if "orig" in n and "size" in n:
        return "orig_size"
    if "label" in n:
        return "labels"
    if "coord" in n or "point" in n:
        return "coords"
    if "embed" in n:
        return "embeddings"
    return None


def np_dtype(ort_type: str):
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(double)": np.float64}.get(ort_type)


def to_original(logits: np.ndarray, h: int, w: int, newh: int, neww: int, size: int) -> np.ndarray:
    """לוגיטים של מסכה אחת -> bool בגודל התמונה המקורית (h, w). מזהה פלט בגודל התמונה המוקטנת, או ריבוע (SAM מרופד)."""
    mn, mx = float(logits.min()), float(logits.max())
    thr = 0.5 if (mn >= 0.0 and mx <= 1.0) else 0.0       # חלק מהייצואים כבר מחילים sigmoid
    lg = logits.astype(np.float32)
    if lg.shape == (newh, neww):
        pass
    elif lg.shape[0] == lg.shape[1]:
        lg = cv2.resize(lg, (size, size), interpolation=cv2.INTER_LINEAR)[:newh, :neww]
    else:
        raise SpikeError(f"unexpected mask output shape {logits.shape} (resized image {newh}x{neww}, size {size})")
    if lg.shape != (h, w):
        lg = cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR)
    return lg > thr


# ------------------------------------------------------------------ מועמד
class Candidate:
    def __init__(self, name: str, encoder: Path, decoder: Path, threads: int, norm: str, size: int, resize: str):
        self.name, self.encoder_path, self.decoder_path = name, encoder, decoder
        self.threads, self.norm, self.resize = threads, norm, resize
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        try:
            t = time.perf_counter()
            self.enc = ort.InferenceSession(str(encoder), so, providers=["CPUExecutionProvider"])
            self.enc_load_s = time.perf_counter() - t
            t = time.perf_counter()
            self.dec = ort.InferenceSession(str(decoder), so, providers=["CPUExecutionProvider"])
            self.dec_load_s = time.perf_counter() - t
        except Exception as e:                       # onnxruntime לא מגדיר סוג שגיאה יחיד (קובץ פגום, אופרטור חסר)
            raise SpikeError(f"{name}: onnxruntime could not load the model: {e}") from e
        self._resolve_contract(size)

    def _resolve_contract(self, size_default: int):
        ins = self.enc.get_inputs()
        if len(ins) != 1:
            raise SpikeError(f"{self.name}: encoder has {len(ins)} inputs {[i.name for i in ins]}; only one image input is supported")
        self.enc_in = ins[0]
        shp = self.enc_in.shape
        if len(shp) == 4 and isinstance(shp[2], int) and isinstance(shp[3], int):
            if shp[2] != shp[3]:
                raise SpikeError(f"{self.name}: non-square encoder input {shp} is not supported")
            self.size = shp[2]
        else:
            self.size = size_default
        self.enc_dtype = np_dtype(self.enc_in.type) or np.float32
        self.enc_out_names = [o.name for o in self.enc.get_outputs()]

        self.dec_roles: dict[str, str] = {}
        unknown = []
        for i in self.dec.get_inputs():
            role = "emb:" + i.name if i.name in self.enc_out_names else role_of(i.name)
            if role is None:
                unknown.append(i)
            else:
                self.dec_roles[i.name] = role
        has_emb = any(r == "embeddings" or r.startswith("emb:") for r in self.dec_roles.values())
        if unknown and len(unknown) == 1 and len(self.enc_out_names) == 1 and not has_emb:
            self.dec_roles[unknown[0].name] = "emb:" + self.enc_out_names[0]   # שם שונה, אותו תפקיד
            unknown = []
        if unknown:
            raise SpikeError(self._contract_msg(f"decoder inputs not recognized: {[i.name for i in unknown]}"))
        got = set(self.dec_roles.values())
        missing = [r for r in ("coords", "labels") if r not in got]
        if missing:
            raise SpikeError(self._contract_msg(f"decoder has no input for: {missing}"))

        outs = self.dec.get_outputs()
        names = [o.name for o in outs]
        masks = [i for i, n in enumerate(names) if n.lower() == "masks"] or \
                [i for i, n in enumerate(names) if "mask" in n.lower() and "low" not in n.lower()] or \
                [i for i, o in enumerate(outs) if len(o.shape) == 4]
        if not masks:
            raise SpikeError(self._contract_msg("no mask output found in decoder"))
        self.mask_out = masks[0]
        iou = [i for i, n in enumerate(names) if "iou" in n.lower() or "score" in n.lower()]
        self.iou_out = iou[0] if iou else None

    def _contract_msg(self, why: str) -> str:
        fmt = lambda xs: [(x.name, x.type, x.shape) for x in xs]
        return (f"{self.name}: {why}. Contract this script cannot map - send this to Claude:\n"
                f"  encoder inputs : {fmt(self.enc.get_inputs())}\n  encoder outputs: {fmt(self.enc.get_outputs())}\n"
                f"  decoder inputs : {fmt(self.dec.get_inputs())}\n  decoder outputs: {fmt(self.dec.get_outputs())}")

    # ---- prepare
    def prepare(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        x, scale, newh, neww = preprocess(rgb, self.size, self.norm, self.resize)
        t1 = time.perf_counter()
        try:
            outs = self.enc.run(None, {self.enc_in.name: x.astype(self.enc_dtype)})
        except Exception as e:
            raise SpikeError(self._contract_msg(f"encoder run failed ({e}); try --norm/--size, or the input layout differs")) from e
        t2 = time.perf_counter()
        return {"emb": dict(zip(self.enc_out_names, outs)), "scale": scale, "newh": newh, "neww": neww,
                "h": rgb.shape[0], "w": rgb.shape[1], "pre_s": t1 - t0, "enc_s": t2 - t1}

    # ---- propose
    def propose(self, ctx: dict, points, box) -> dict:
        t0 = time.perf_counter()
        coords, labels = encode_prompts(points, box, ctx["scale"])
        feed = {}
        for i in self.dec.get_inputs():
            role, dt = self.dec_roles[i.name], np_dtype(i.type) or np.float32
            if role.startswith("emb:"):
                feed[i.name] = ctx["emb"][role[4:]]
            elif role == "embeddings":
                feed[i.name] = next(iter(ctx["emb"].values()))
            elif role == "coords":
                feed[i.name] = coords.astype(dt)
            elif role == "labels":
                feed[i.name] = labels.astype(dt)
            elif role == "mask_input":
                dims = [d if isinstance(d, int) else df for d, df in zip(i.shape, (1, 1, 256, 256))]
                feed[i.name] = np.zeros(dims, dt)
            elif role == "has_mask":
                feed[i.name] = np.zeros(i.shape if all(isinstance(d, int) for d in i.shape) else (1,), dt)
            elif role == "orig_size":
                feed[i.name] = np.array([ctx["newh"], ctx["neww"]], dt)    # גודל התמונה המוקטנת, לא המלאה
        t1 = time.perf_counter()
        try:
            outs = self.dec.run(None, feed)
        except Exception as e:
            raise SpikeError(self._contract_msg(f"decoder run failed ({e})")) from e
        t2 = time.perf_counter()
        raw = np.asarray(outs[self.mask_out])
        raw = raw.reshape((-1,) + raw.shape[-2:])                          # K x h x w
        masks = [to_original(m, ctx["h"], ctx["w"], ctx["newh"], ctx["neww"], self.size) for m in raw]
        scores = None if self.iou_out is None else [round(float(s), 4) for s in np.asarray(outs[self.iou_out]).ravel()]
        t3 = time.perf_counter()
        best = int(np.argmax(scores)) if scores and len(scores) == len(masks) else 0
        return {"masks": masks, "scores": scores, "best": best, "dec_s": t2 - t1, "post_s": (t1 - t0) + (t3 - t2)}


# ------------------------------------------------------------------ עומס רקע (Inpaint)
class LamaLoad:
    """Big LaMa ברקע על 3 threads (ADR-011), כדי למדוד את הבחירה בזמן Inpaint. שמות הטנסורים נקראים מהקובץ (ADR-005)."""

    def __init__(self, path: Path, side: int = 384):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        names = [i.name for i in self.sess.get_inputs()]
        rng = np.random.default_rng(0)
        mask = np.zeros((1, 1, side, side), np.float32)
        mask[..., side // 3: 2 * side // 3, side // 3: 2 * side // 3] = 1
        self.feed = {names[0]: rng.random((1, 3, side, side), dtype=np.float32), names[1]: mask}
        self.stop, self.runs = threading.Event(), 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            self.sess.run(None, self.feed)
            self.runs += 1

    def __enter__(self):
        self.thread.start()
        time.sleep(1.0)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


# ------------------------------------------------------------------ מדידה
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return round(float(np.logical_and(a, b).sum() / u), 4) if u else 1.0


def overlay(rgb, res: dict, box, click):
    out = rgb.copy()
    best = res["masks"][res["best"]]
    out[best] = (0.5 * out[best] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    palette = [(255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
    for k, m in enumerate(res["masks"]):
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, (255, 255, 255) if k == res["best"] else palette[k % 4], 1)
    if box:
        cv2.rectangle(out, box[:2], box[2:], (0, 255, 255), 1)
    if click:
        cv2.circle(out, (int(click[0]), int(click[1])), 4, (0, 255, 0), -1)
    return out


def components(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)


def measure_prepare(cand: Candidate, rgb: np.ndarray, repeats: int, first_too: bool) -> dict:
    res = {}
    with RssSampler() as rs:
        if first_too:
            c = cand.prepare(rgb)
            res["first_ms"] = round((c["pre_s"] + c["enc_s"]) * 1000, 1)
        pre, enc = [], []
        for _ in range(repeats):
            c = cand.prepare(rgb)
            pre.append(c["pre_s"]); enc.append(c["enc_s"])
    res.update(warm_ms=round(median_ms(pre) + median_ms(enc), 1), pre_ms=median_ms(pre), enc_ms=median_ms(enc),
               peak_rss_mb=rs.peak_mb)
    return res


def measure_proposals(cand: Candidate, images: dict, cases: list, repeats: int, grabcut, out_dir: Path | None) -> list[dict]:
    ctxs = {name: cand.prepare(rgb) for name, rgb in images.items()}
    rows = []
    for case in cases:
        rgb, ctx, box = images[case["image"]], ctxs[case["image"]], tuple(case["box"])
        click = tuple(case.get("point") or ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2))
        kinds = {"click": ([(click[0], click[1], True)], None), "box": ([], box)}
        best_masks = {}
        for kind, (pts, bx) in kinds.items():
            cand.propose(ctx, pts, bx)                                      # חימום, לא נספר
            dec, post, res = [], [], None
            for _ in range(repeats):
                res = cand.propose(ctx, pts, bx)
                dec.append(res["dec_s"]); post.append(res["post_s"])
            m = res["masks"][res["best"]]
            best_masks[kind] = m
            row = {"case": case["name"], "kind": kind, "dec_ms": median_ms(dec), "post_ms": median_ms(post),
                   "total_ms": round(median_ms(dec) + median_ms(post), 1), "scores": res["scores"], "best": res["best"],
                   "area_frac": round(float(m.mean()), 4), "components": components(m),
                   "alt_area_frac": [round(float(x.mean()), 4) for x in res["masks"]]}
            if grabcut is not None and kind == "box":
                row["iou_vs_grabcut"] = grabcut(case["image"], rgb, box, m)
            rows.append(row)
            if out_dir is not None:
                imwrite_rgb(out_dir / f"{case['name']}_{kind}.png", overlay(rgb, res, bx, click if kind == "click" else None))
        rows[-2]["iou_click_vs_box"] = rows[-1]["iou_click_vs_box"] = iou(best_masks["click"], best_masks["box"])
    return rows


def make_grabcut_comparer():
    """IoU מול GrabCutProvider (אינפורמטיבי בלבד, לא אמת-קרקע). None אם fixpic לא זמין."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from fixpic.select import Prompts
        from fixpic.select.grabcut import GrabCutProvider
    except Exception:
        return None
    prov, ctxs = GrabCutProvider(), {}

    def run(image_name, rgb, box, mask):
        if image_name not in ctxs:
            ctxs[image_name] = prov.prepare(rgb)
        prop = prov.propose(ctxs[image_name], Prompts(box=box))
        if prop is None:
            return None
        ref = np.zeros(mask.shape, bool)
        x0, y0, x1, y1 = prop.rect
        ref[y0:y1, x0:x1] = prop.window > 0
        return iou(mask, ref)
    return run


def measure_run(cand: Candidate, images, cases, a, grabcut, out_dir, big_rgb, label: str, rss_before_load) -> dict:
    run = {"threads": cand.threads, "condition": label, "enc_size": cand.size,
           "load_s": {"encoder": round(cand.enc_load_s, 2), "decoder": round(cand.dec_load_s, 2)}}
    after_load = rss_mb()
    person = images[cases[0]["image"]]
    run["prepare_person"] = measure_prepare(cand, person, a.prepare_repeats, first_too=True)
    if big_rgb is not None:
        run["prepare_24mp"] = measure_prepare(cand, big_rgb, max(1, a.prepare_repeats - 1), first_too=False)
        c = cand.prepare(person)
        res = cand.propose(c, [], tuple(cases[0]["box"]))
        small = (res["masks"][res["best"]] * 255).astype(np.uint8)
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = cv2.resize(small, (big_rgb.shape[1], big_rgb.shape[0]), interpolation=cv2.INTER_LINEAR) > 127
            t.append(time.perf_counter() - t0)
        run["fullres_mask_24mp_ms"] = median_ms(t)
    run["propose"] = measure_proposals(cand, images, cases, a.propose_repeats, grabcut, out_dir)
    peaks = [run[k]["peak_rss_mb"] for k in ("prepare_person", "prepare_24mp") if k in run and run[k]["peak_rss_mb"]]
    run["memory"] = {"rss_before_load_mb": rss_before_load, "rss_after_load_mb": after_load,
                     "peak_prepare_mb": max(peaks) if peaks else None,
                     "extra_peak_mb": (max(peaks) - rss_before_load) if (peaks and rss_before_load) else None}
    return run


def evaluate_gate(runs: list[dict]) -> dict:
    """מספרי השער בלבד. איכות, זיכרון ורישיון נשארים להכרעה ידנית."""
    def prep_s(r):
        p = r.get("prepare_24mp") or r["prepare_person"]
        return p["warm_ms"] / 1000

    def prop_s(r):
        return max(x["total_ms"] for x in r["propose"]) / 1000
    idle = [r for r in runs if r["condition"] == "idle"]
    busy = [r for r in runs if r["condition"] == "contention"]
    most_constrained = min(runs, key=lambda r: r["threads"])
    gate = {"prepare_s_best": round(min(prep_s(r) for r in idle), 2),
            "propose_s_worst_at_fewest_threads": round(prop_s(min(idle, key=lambda r: r["threads"])), 3),
            "extra_peak_mb": runs[0]["memory"]["extra_peak_mb"],     # הריצה הראשונה: הבסיס הנקי, לפני שארית של טעינה קודמת
            "threads_most_constrained": most_constrained["threads"]}
    if busy:
        gate["propose_s_worst_during_inpaint"] = round(max(prop_s(r) for r in busy), 3)
    gate["prepare_ok"] = gate["prepare_s_best"] <= GATE_PREPARE_S
    gate["propose_ok"] = max(gate["propose_s_worst_at_fewest_threads"],
                             gate.get("propose_s_worst_during_inpaint", 0)) <= GATE_PROPOSE_S
    gate["manual"] = ["quality 3/3 by eye (PNG files)", "memory vs 7.9GB machine", "license (code, weights, training data)"]
    return gate


def describe_files(c: Candidate) -> dict:
    return {k: {"path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1)} for k, p in
            (("encoder", c.encoder_path), ("decoder", c.decoder_path))}


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_CASES
    data = json.loads(Path(path).read_text("utf-8"))
    return [{"name": d["name"], "image": d["image"], "box": tuple(d["box"]), "point": d.get("point")} for d in data]


def resolve_image(name: str, images_dir: Path) -> Path:
    p = Path(name)
    return p if p.is_file() else images_dir / f"{name}.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "ENCODER", "DECODER"), required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[4, 1])
    ap.add_argument("--contention", nargs="?", const=str(ROOT / "models" / "big-lama" / "inpainter.onnx"), default=None)
    ap.add_argument("--norm", choices=["sam", "unit", "none"], default="sam", help="input normalization of the encoder")
    ap.add_argument("--resize", choices=["area", "linear"], default="area")
    ap.add_argument("--size", type=int, default=1024, help="encoder side if the file has a dynamic input shape")
    ap.add_argument("--images-dir", default=str(ROOT / "samples"))
    ap.add_argument("--cases", help="JSON: [{name, image, box:[x0,y0,x1,y1], point:[x,y]?}]")
    ap.add_argument("--prepare-repeats", type=int, default=3)
    ap.add_argument("--propose-repeats", type=int, default=5)
    ap.add_argument("--no-24mp", action="store_true")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "m4_out" / "model_spike"))
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        cases = load_cases(a.cases)
        images_dir = Path(a.images_dir)
        images = {}
        for c in cases:
            if c["image"] not in images:
                images[c["image"]] = imread_rgb(resolve_image(c["image"], images_dir))
        for _, enc, dec in a.candidate:
            check_model_file(Path(enc)); check_model_file(Path(dec))
        lama = None
        if a.contention:
            lama = Path(a.contention)
            check_model_file(lama)
    except SpikeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    env = env_info()
    print(f"CPU: {env['cpu']} | cores {env.get('physical_cores')}/{env['logical_cores']} | RAM {env.get('ram_gb')}GB")
    if env.get("on_battery"):
        print("WARNING: running on battery; plug in for stable timings")
    if (env.get("load_before_pct") or 0) > 25:
        print(f"WARNING: CPU load before start {env['load_before_pct']}%; close other programs")
    base = images[cases[0]["image"]]
    big = None if a.no_24mp else cv2.resize(base, (6000, 4000), interpolation=cv2.INTER_CUBIC)
    grabcut = make_grabcut_comparer()
    results = {"env": env, "gate_targets": {"propose_s": GATE_PROPOSE_S, "prepare_s": GATE_PREPARE_S}, "candidates": []}

    for name, enc, dec in a.candidate:
        enc, dec = Path(enc), Path(dec)
        entry = {"name": name, "license": license_info(enc.parent), "runs": []}
        if not a.no_hash:
            entry["sha256"] = {"encoder": sha256_of(enc), "decoder": sha256_of(dec)}
        try:
            for th in a.threads:
                print(f"\n== {name}: threads={th}")
                before = rss_mb()
                cand = Candidate(name, enc, dec, th, a.norm, a.size, a.resize)
                entry.setdefault("files", describe_files(cand))
                entry.setdefault("io", {"encoder_input": [cand.enc_in.name, cand.enc_in.shape, cand.size],
                                        "decoder_roles": cand.dec_roles})
                print(f"   loaded: enc {cand.enc_load_s:.1f}s, dec {cand.dec_load_s:.1f}s, side {cand.size}, "
                      f"RSS +{(rss_mb() - before) if before else '?'}MB")
                for label, ctx_mgr in (("idle", None), ("contention", lama)):
                    if label == "contention" and lama is None:
                        continue
                    if ctx_mgr is None:
                        run = measure_run(cand, images, cases, a, grabcut, out / name / f"t{th}" if label == "idle" else None, big, label, before)
                    else:
                        with LamaLoad(ctx_mgr):
                            run = measure_run(cand, images, cases, a, grabcut, None, big, label, before)
                    entry["runs"].append(run)
                    p = run["prepare_person"]
                    print(f"   [{label}] prepare 0.4MP: first {p.get('first_ms')}ms, warm {p['warm_ms']}ms "
                          f"(pre {p['pre_ms']} + enc {p['enc_ms']})")
                    if "prepare_24mp" in run:
                        q = run["prepare_24mp"]
                        print(f"   [{label}] prepare 24MP: warm {q['warm_ms']}ms (pre {q['pre_ms']} + enc {q['enc_ms']}), "
                              f"peak RSS {q['peak_rss_mb']}MB; full-res mask {run['fullres_mask_24mp_ms']}ms")
                    for r in run["propose"]:
                        extra = f" iou_gc={r['iou_vs_grabcut']}" if r.get("iou_vs_grabcut") is not None else ""
                        print(f"   [{label}] {r['case']:24} {r['kind']:5} {r['total_ms']:7.1f}ms "
                              f"(dec {r['dec_ms']}) area {r['area_frac']:.3f} comps {r['components']}{extra}")
                del cand
            entry["gate"] = evaluate_gate(entry["runs"])
        except SpikeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            entry["error"] = str(e)
        results["candidates"].append(entry)

    (out / "model_spike_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), "utf-8")
    print("\n===== gate (numeric part) =====")
    for e in results["candidates"]:
        g = e.get("gate")
        if not g:
            print(f"{e['name']}: NOT MEASURED ({e.get('error', '').splitlines()[0] if e.get('error') else ''})")
            continue
        print(f"{e['name']}: prepare {g['prepare_s_best']}s (<= {GATE_PREPARE_S}) {'OK' if g['prepare_ok'] else 'FAIL'} | "
              f"propose {max(g['propose_s_worst_at_fewest_threads'], g.get('propose_s_worst_during_inpaint', 0))}s "
              f"(<= {GATE_PROPOSE_S}) {'OK' if g['propose_ok'] else 'FAIL'} | extra peak RSS {g['extra_peak_mb']}MB")
        print(f"   still manual: {', '.join(g['manual'])}")
        print(f"   license files next to model: {e['license']['files_next_to_model'] or 'NONE FOUND'} (verify at the source)")
    print(f"\nResults: {out / 'model_spike_results.json'}  | images for visual check: {out}")
    return 0 if all("gate" in e for e in results["candidates"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
