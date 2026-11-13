"""CLI לבדיקת המנוע (M2.7): `python -m fixpic תמונה מסכה [-o פלט]`. ללא UI; עובר דרך ה-JobScheduler כמו האפליקציה.

מסכה: כל פיקסל מעל --mask-threshold (ברירת מחדל 0) יוסר. תמונה: RGB 8 סיביות; ערוץ Alpha נזרק (טיפול ב-Alpha
יתווסף עם השמירה באפליקציה). נתיבים עם עברית נתמכים (קריאה/כתיבה דרך numpy ולא דרך cv2.imread).

קודי יציאה: 0 הצלחה; 1 שגיאה אחרת; 2 קלט לא תקין (קובץ, גודל, ארגומנטים); 3 מודל לא נטען; 4 מסכה ריקה;
5 כשל הרצה או זיכרון; 130 בוטל (Ctrl+C)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .engine import JobScheduler, JobState
from .errors import (CancelledError, EmptyMaskError, FixPicError, InferenceError, InvalidInputError,
                     ModelLoadError, OutOfMemoryError)
from .models import ModelManager
from .models.ort_model import OrtModel
from .pipeline import Pipeline, PipelineConfig

DEFAULT_MODELS = Path(__file__).resolve().parents[2] / "models"
EXIT_OK, EXIT_OTHER, EXIT_INPUT, EXIT_MODEL, EXIT_EMPTY, EXIT_RUN, EXIT_CANCEL = 0, 1, 2, 3, 4, 5, 130


def exit_code(e: BaseException) -> int:
    if isinstance(e, CancelledError):
        return EXIT_CANCEL
    if isinstance(e, InvalidInputError):
        return EXIT_INPUT
    if isinstance(e, ModelLoadError):
        return EXIT_MODEL
    if isinstance(e, EmptyMaskError):
        return EXIT_EMPTY
    if isinstance(e, (InferenceError, OutOfMemoryError)):
        return EXIT_RUN
    return EXIT_OTHER


def read_image(path: Path) -> np.ndarray:
    """RGB uint8 [H,W,3]. כשל קריאה = InvalidInputError."""
    return _decode(path, cv2.IMREAD_COLOR, "תמונה")[..., ::-1].copy()


def read_mask(path: Path, threshold: int = 0) -> np.ndarray:
    """uint8 [H,W], 255 = להסרה."""
    g = _decode(path, cv2.IMREAD_GRAYSCALE, "מסכה")
    return (g > threshold).astype(np.uint8) * 255


def _decode(path: Path, flag: int, what: str) -> np.ndarray:
    try:
        buf = np.fromfile(str(path), np.uint8)
    except OSError as e:
        raise InvalidInputError(f"לא ניתן לקרוא {what}: {path} ({e})") from e
    img = cv2.imdecode(buf, flag) if buf.size else None
    if img is None:
        raise InvalidInputError(f"{what} פגומה או בפורמט לא נתמך: {path}")
    return img


def write_image(path: Path, rgb: np.ndarray) -> None:
    """כתיבה אטומית (קובץ זמני ואז החלפה): כשל באמצע לא משאיר פלט חלקי."""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, rgb[..., ::-1])
    if not ok:
        raise InvalidInputError(f"פורמט פלט לא נתמך: {ext}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        buf.tofile(str(tmp))
        os.replace(tmp, path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise InvalidInputError(f"כתיבת הפלט נכשלה: {path} ({e})") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fixpic", description="הסרת אלמנטים מתמונה (בדיקת המנוע)")
    p.add_argument("image", type=Path)
    p.add_argument("mask", type=Path, help="תמונת מסכה; לבן = להסרה")
    p.add_argument("-o", "--output", type=Path, help="ברירת מחדל: <תמונה>_out.png ליד הקלט")
    p.add_argument("--size", type=int, default=384, help="bucket: 256 (אינטראקטיבי), 384 (רגיל), 512 (איכות)")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--backend", choices=("auto", "ort", "ov"), default="auto",
                   help="auto = OpenVINO עם fallback ל-ORT (ADR-011)")
    p.add_argument("--model", default="big-lama")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--memory-limit-mb", type=int, help="תקרת זיכרון להערכה מראש (כולל המודל)")
    p.add_argument("--mask-threshold", type=int, default=0)
    p.add_argument("--dilate", type=int, help="הרחבת מסכה בפיקסלים (0 = כבוי)")
    p.add_argument("--feather", type=int, help="Feathering בפיקסלים (0 = כבוי)")
    p.add_argument("--progress", action="store_true", help="הדפסת התקדמות (ברירת מחדל: רק בטרמינל)")
    p.add_argument("--json", action="store_true", help="סיכום כ-JSON ב-stdout")
    return p


def _load_model(a):
    mgr = ModelManager(a.models_dir)
    if a.backend == "auto":
        return mgr.load(a.model, threads=a.threads)
    info = mgr.manifest(a.model).info
    if a.backend == "ort":
        return mgr.load(a.model, threads=a.threads, model=OrtModel(info))
    try:
        from .models.ov_model import OvModel
    except ImportError as e:  # pragma: no cover
        raise ModelLoadError(f"OpenVINO לא מותקן: {e}") from e
    return mgr.load(a.model, threads=a.threads, model=OvModel(info))


def _wait(job) -> None:
    """ממתין לעבודה; Ctrl+C מבטל אותה (ביטול שני: יציאה מיידית)."""
    try:
        while not job.wait(0.2):
            pass
    except KeyboardInterrupt:
        job.cancel()
        print("מבטל...", file=sys.stderr)
        while not job.wait(0.2):
            pass


def run(a: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    image = read_image(a.image)
    mask = read_mask(a.mask, a.mask_threshold)
    out_path = a.output or a.image.with_name(a.image.stem + "_out.png")
    if out_path.resolve() == a.image.resolve():
        raise InvalidInputError("נתיב הפלט זהה לתמונת הקלט; הקלט לא נדרס")
    overrides = {}
    if a.dilate is not None:
        overrides["mask_dilate_px"] = a.dilate
    if a.feather is not None:
        overrides["feather_px"] = a.feather
    if a.memory_limit_mb is not None:
        overrides["memory_limit_mb"] = a.memory_limit_mb
    model = _load_model(a)
    t_load = time.perf_counter() - t0
    pipeline = Pipeline(model, PipelineConfig(**overrides))
    show = a.progress or sys.stderr.isatty()

    def on_progress(job, p):
        if show:
            print(f"  {p.stage}: {p.done}/{p.total}", file=sys.stderr)

    sched = JobScheduler(pipeline).start(warmup=False)
    try:
        t1 = time.perf_counter()
        job = sched.submit(image, mask, a.size, on_progress=on_progress)
        _wait(job)
        res = job.result(0)          # CancelledError / שגיאת העבודה
        t_run = time.perf_counter() - t1
    finally:
        sched.shutdown(wait=False)
    write_image(out_path, res.image)
    return {"output": str(out_path), "backend": getattr(model, "backend", ""),
            "fallback_reason": getattr(model, "fallback_reason", None), "size": a.size,
            "size_used": res.size_used, "groups": res.groups, "model_calls": res.model_calls,
            "oom_retries": res.oom_retries, "image_hw": list(image.shape[:2]),
            "mask_pixels": int(res.effective_mask.sum()),
            "load_s": round(t_load, 2), "run_s": round(t_run, 2)}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        info = run(a)
    except FixPicError as e:
        print(f"שגיאה: {e}", file=sys.stderr)
        return exit_code(e)
    except KeyboardInterrupt:
        print("בוטל", file=sys.stderr)
        return EXIT_CANCEL
    if a.json:
        print(json.dumps(info, ensure_ascii=False))
    else:
        print(f"נשמר: {info['output']}  backend={info['backend'] or '-'}  חלונות={info['model_calls']}  "
              f"קבוצות={info['groups']}  טעינה={info['load_s']}s  הרצה={info['run_s']}s")
    return EXIT_OK
