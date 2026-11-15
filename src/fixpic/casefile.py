"""מקרה בדיקה (M3.7.1): תיקייה עם תמונת המקור, המסכה שהוגשה, התוצאה ופרמטרי ה-Job, לשחזור בעיית איכות ולמדידה חוזרת.

מבנה: `original.png`, `mask.png` (0/255), `result.png` (אופציונלי), `reference.png` (אופציונלי, למשל תוצאת אתר להשוואה),
`case.json`. מקור שגדול מדי לשמירה כפולה אפשר לתאר ב-`case.json` בשדה `original_ref` (נתיב יחסי) במקום `original.png`.
הנתיבים מטופלים כ-Unicode (תיקיות בעברית)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .errors import InvalidInputError
from .imgio import load_image, save_image

SCHEMA = 1


def _write_mask(path: Path, mask: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    if not ok:
        raise InvalidInputError("קידוד המסכה נכשל")
    path.write_bytes(buf.tobytes())


def read_mask(path: str | Path) -> np.ndarray:
    m = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise InvalidInputError(f"לא ניתן לקרוא מסכה: {path}")
    return np.where(m > 127, 255, 0).astype(np.uint8)


def save_case(folder: str | Path, *, mask: np.ndarray, original_rgb: np.ndarray | None = None,
              result_rgb: np.ndarray | None = None, reference_rgb: np.ndarray | None = None,
              params: dict | None = None, note: str = "", original_ref: str | None = None) -> Path:
    """שומר מקרה בתיקייה (נוצרת אם חסרה). חייב `original_rgb` או `original_ref`. מחזיר את התיקייה."""
    if original_rgb is None and original_ref is None:
        raise InvalidInputError("חסר מקור: original_rgb או original_ref")
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape[:2]
    if original_rgb is not None and original_rgb.shape[:2] != (h, w):
        raise InvalidInputError("מידות המסכה והמקור שונות")
    if original_rgb is not None:
        save_image(d / "original.png", original_rgb)
    if result_rgb is not None:
        save_image(d / "result.png", result_rgb)
    if reference_rgb is not None:
        save_image(d / "reference.png", reference_rgb)
    _write_mask(d / "mask.png", mask)
    meta = {"schema": SCHEMA, "width": int(w), "height": int(h), "mask_pixels": int((mask > 0).sum()),
            "params": params or {}, "note": note}
    if original_ref is not None:
        meta["original_ref"] = original_ref
    (d / "case.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def load_case(folder: str | Path) -> dict:
    """מחזיר {'meta', 'rgb', 'mask', 'result', 'reference'} (result/reference None אם חסרים)."""
    d = Path(folder)
    meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise InvalidInputError(f"סכמת מקרה לא נתמכת: {meta.get('schema')}")
    src = d / "original.png" if (d / "original.png").exists() else (d / meta["original_ref"]).resolve()
    out = {"meta": meta, "rgb": load_image(src).rgb, "mask": read_mask(d / "mask.png"), "result": None, "reference": None}
    for k, n in (("result", "result.png"), ("reference", "reference.png")):
        if (d / n).exists():
            out[k] = load_image(d / n).rgb
    if out["mask"].shape != out["rgb"].shape[:2]:
        raise InvalidInputError("מידות המסכה והמקור במקרה אינן שוות")
    return out


def derive_mask(original: np.ndarray, edited: np.ndarray, *, threshold: int = 18, min_area: int = 200,
                dilate: int = 3) -> np.ndarray:
    """משחזר מסכה מהפרש בין מקור לתוצאה (למקרים שבהם המסכה המקורית אבדה, כמו תוצאת אתר). הפרש מרבי בין ערוצים אחרי
    טשטוש קל, סף, סגירה, הסרת רכיבים קטנים והרחבה. הערכה בלבד: אזורים שהתוצאה שינתה מעט ייתכן שחסרים."""
    if original.shape != edited.shape:
        raise InvalidInputError("מידות המקור והתוצאה שונות")
    a = cv2.GaussianBlur(original, (0, 0), 1.2).astype(np.int16)
    b = cv2.GaussianBlur(edited, (0, 0), 1.2).astype(np.int16)
    diff = np.abs(a - b).max(axis=2)
    m = (diff > threshold).astype(np.uint8)
    k = max(3, (min(m.shape) // 150) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area)
    m = keep[lab].astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnts, -1, 1, thickness=cv2.FILLED)         # חורים פנימיים (שבהם המילוי דומה למקור) נסגרים
    if dilate > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    return (m * 255).astype(np.uint8)
