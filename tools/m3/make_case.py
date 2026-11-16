"""M3.7.1: יוצר מקרה בדיקה מקובצי תמונה: מקור + תוצאה (של הכלי או של אתר) -> מסכה משוחזרת מההפרש.

הרצה: `python tools/m3/make_case.py ORIGINAL EDITED -o cases/NAME [--reference REF] [--note TEXT] [--ref-original]`
`--ref-original`: לא מעתיק את המקור לתיקייה (קובץ גדול), אלא רושם נתיב יחסי אליו ב-`case.json`.
המסכה הנוצרת היא הערכה מההפרש; יש לבדוק אותה בעין (נשמרת גם `mask_preview.jpg`: המקור עם האזור באדום)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fixpic.casefile import derive_mask, save_case  # noqa: E402
from fixpic.imgio import load_image, save_image  # noqa: E402


def fit_edited(orig: np.ndarray, edited: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """M3.7.11: תוצאת אתר היא לרוב קטנה מהמקור (למשל 682x1024 מול 3456x5184). אם יחס הגובה-רוחב זהה (עד `tol`), מגדילים
    את התוצאה לגודל המקור (cubic) לצורך חישוב המסכה בלבד; `--reference` נשמר בגודלו המקורי. אחרת: שגיאה ברורה."""
    if edited.shape[:2] == orig.shape[:2]:
        return edited
    (h, w), (eh, ew) = orig.shape[:2], edited.shape[:2]
    if abs((w / h) / (ew / eh) - 1) > tol:
        raise SystemExit(f"יחס הגובה-רוחב שונה: מקור {w}x{h}, תוצאה {ew}x{eh}. ודא שזו אותה תמונה (בלי חיתוך/סיבוב; ב-EXIF הכלי מתקן כיוון).")
    print(f"התוצאה ({ew}x{eh}) הוגדלה ל-{w}x{h} לחישוב המסכה בלבד")
    return cv2.resize(edited, (w, h), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("edited")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--note", default="")
    ap.add_argument("--ref-original", action="store_true")
    ap.add_argument("--threshold", type=int, default=18)
    a = ap.parse_args()
    orig, edited = load_image(a.original).rgb, load_image(a.edited).rgb
    edited = fit_edited(orig, edited)
    mask = derive_mask(orig, edited, threshold=a.threshold)
    out = Path(a.out)
    ref = load_image(a.reference).rgb if a.reference else None
    save_case(out, mask=mask, original_rgb=None if a.ref_original else orig, result_rgb=None, reference_rgb=ref,
              params={"derived_from": Path(a.edited).name}, note=a.note,
              original_ref=os.path.relpath(a.original, out) if a.ref_original else None)
    prev = orig.copy()
    prev[mask > 0] = (0.5 * prev[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    save_image(out / "mask_preview.jpg", prev, quality=80)
    ys, xs = np.nonzero(mask)
    print(f"{out}: {orig.shape[1]}x{orig.shape[0]}, mask {int((mask > 0).sum())} px "
          f"({100 * (mask > 0).mean():.1f}%), bbox x {xs.min()}..{xs.max()} y {ys.min()}..{ys.max()}")


if __name__ == "__main__":
    main()
