"""M3.7.8: שחזור טקסטורה באזור מולא גדול (הסתרת "מריחה").

מודל מילוי שרץ בהקטנה (פי 4-16 באזורים גדולים) מחזיר מילוי חלק: אנרגיית התדר הגבוה שלו נמוכה בהרבה מזו של הסביבה
(`quality.texture_ratio` 0.3 מול ייחוס 1.0). כאן מוסיפים גרעין תדר-גבוה (רעש מסונן ברוחב `SIGMA_HP`, זהה למסנן שבו
המדד מודד) בתוך המסכה בלבד, בעוצמה שמותאמת **מקומית** לסביבה:

1. אנרגיית התדר הגבוה (`hp^2`, על הבהירות) נמדדת בפסים ובמבנה מוקטן (בלוקים של `f` על `f`), בפנים המולא ובסביבה.
2. יעד: `E_t` = אנרגיית הסביבה (טבעת סביב הגבול) מורחבת פנימה בהתכנסות מנורמלת (טשטוש של האנרגיה חלקי טשטוש
   המשקל), כך שצד עם דשא מקבל גרעין של דשא וצד עם קיר חלק כמעט לא מקבל; רחוק מהטבעת מתכנסים לממוצע הכללי.
3. חוסר: `E_t - E_i` (מקומי, ללא ערכים שליליים) הוא השונות שחסרה; רעש בלתי-תלוי מוסיף שונות בריבוע, ולכן
   המשרעת היא `sqrt(חוסר) * strength`. היכן שהמילוי כבר עשיר (מבנה, קווים) אין חוסר ואין רעש.
4. מגן: אם יחס הטקסטורה הכללי (אותה נוסחה כמו `quality.texture_ratio`) לפחות `guard`, לא נוגעים בכלום.

רק פיקסלים של מסכת הקבוצה משתנים (מחוץ לה: זהה לבייט). הרעש דטרמיניסטי (זרע + תיבת הקבוצה + אינדקס פס) ומוסף
זהה לשלושת הערוצים (גרעין בהירות, בלי רעש צבע). הזיכרון: פסים של `STRIP` שורות ומפות מוקטנות של כ-1.5MP לכל היותר.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..models.base import CancelToken

SIGMA_HP = 1.5      # אותו מסנן כמו `quality._highpass`
SIGMA_PRE = 0.6     # החלקה קלה של הרעש הלבן לפני המסנן: גרעין "צילומי" ולא פיקסל-פיקסל
STRIP = 256         # שורות בפס
HALO = 8            # שורות/עמודות הקשר סביב פס (מעל רדיוס הטשטוש)
_EPS_RMS = 0.25     # כמו `quality._EPS_RMS`
_REDUCED_PX = 1_500_000
_MIN_AMP = 0.05     # מתחת לזה (RMS ברמות אפור) אין מה להוסיף

_norm_cache: list[float] = []


@dataclass(frozen=True)
class TextureInfo:
    applied: bool
    ratio: float = float("nan")   # יחס הטקסטורה לפני (פנים/סביבה), כפי שחושב במגן
    amp_rms: float = 0.0           # RMS של הרעש שנוסף, ברמות אפור (ממוצע משוקלל על הפנים)
    reason: str = ""              # כשלא הופעל: למה


def _grain(n: np.ndarray) -> np.ndarray:
    b = cv2.GaussianBlur(n, (0, 0), SIGMA_PRE)
    return b - cv2.GaussianBlur(b, (0, 0), SIGMA_HP)


def _grain_norm() -> float:
    """סטיית התקן של הגרעין הגולמי (קבוע, נמדד פעם אחת), כדי שגרעין * משרעת = RMS בדיוק של המשרעת."""
    if not _norm_cache:
        n = np.random.default_rng(12345).standard_normal((256, 256), dtype=np.float32)
        _norm_cache.append(float(_grain(n)[16:-16, 16:-16].std()))
    return _norm_cache[0]


def _bsum(a: np.ndarray, f: int, rw: int) -> np.ndarray:
    """סכום בלוקים f על f (עם ריפוד באפסים בקצוות) לפס ברוחב עמודות `rw*f` לכל היותר."""
    n, w = a.shape
    rh = -(-n // f)
    if n != rh * f or w != rw * f:
        a = np.pad(a, ((0, rh * f - n), (0, rw * f - w)))
    return a.reshape(rh, f, rw, f).sum(axis=(1, 3))


def _gauss(a: np.ndarray, s: float) -> np.ndarray:
    """טשטוש גאוסי רחב על מפה חלקה: מעל σ=6 מוקטן קודם (INTER_AREA) ומוגדל אחר כך, כי טשטוש ברזולוציה מלאה עם σ≈100
    עלה כ-0.6 שניות למפה (פי 30 מההקטנה)."""
    k = int(s // 6)
    if k < 2:
        return cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
    h, w = a.shape
    small = cv2.resize(a, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), s / k, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def restore_texture(result: np.ndarray, m_all: np.ndarray, box: tuple[int, int, int, int], g_mask: np.ndarray, *,
                    strength: float = 1.0, guard: float = 0.85, seed: int = 0,
                    cancel: CancelToken | None = None) -> TextureInfo:
    """מוסיף גרעין ל-`result` (במקום) בפיקסלים של `g_mask` (בגודל `box`, בקואורדינטות התמונה).
    `m_all`: המסכה האפקטיבית של כל התמונה (פיקסלים מחוץ לה = המקור שממנו נמדדת הסביבה). מחזיר `TextureInfo`."""
    h, w = result.shape[:2]
    x0, y0, x1, y1 = box
    sel_g = g_mask > 0
    area = int(sel_g.sum())
    if area == 0 or strength <= 0:
        return TextureInfo(False, reason="אין מה לשחזר")
    ring = int(np.clip(round(0.1 * math.sqrt(area)), 8, 96))
    inset = max(2, ring // 4)
    pad = ring + inset + 4
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    ch, cw = cy1 - cy0, cx1 - cx0
    inside = np.zeros((ch, cw), np.uint8)
    inside[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = sel_g
    outside = (m_all[cy0:cy1, cx0:cx1] == 0).astype(np.uint8)
    cancel = cancel or CancelToken()

    # ---- 1. אנרגיית תדר גבוה בבלוקים (f על f), בפסים
    f = max(1, math.ceil(math.sqrt(ch * cw / _REDUCED_PX)))
    rh, rw = -(-ch // f), -(-cw // f)
    s_in = np.zeros((rh, rw), np.float64)
    c_in = np.zeros((rh, rw), np.float64)
    s_out = np.zeros((rh, rw), np.float64)
    c_out = np.zeros((rh, rw), np.float64)
    rows = f * max(1, STRIP // f)
    for r0 in range(0, ch, rows):
        cancel.raise_if_cancelled()
        r1 = min(ch, r0 + rows)
        a, b = max(0, r0 - HALO), min(ch, r1 + HALO)
        luma = cv2.cvtColor(np.ascontiguousarray(result[cy0 + a:cy0 + b, cx0:cx1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
        e = (luma - cv2.GaussianBlur(luma, (0, 0), SIGMA_HP))[r0 - a:r1 - a] ** 2
        i_s, o_s = inside[r0:r1].astype(np.float32), outside[r0:r1].astype(np.float32)
        q = r0 // f
        for dst, arr in ((s_in, e * i_s), (c_in, i_s), (s_out, e * o_s), (c_out, o_s)):
            blk = _bsum(arr, f, rw)
            dst[q:q + blk.shape[0]] += blk

    f2 = float(f * f)
    in_frac = (c_in / f2).astype(np.float32)
    out_frac = (c_out / f2).astype(np.float32)
    mean_in = (s_in / np.maximum(c_in, 1)).astype(np.float32)
    mean_out = (s_out / np.maximum(c_out, 1)).astype(np.float32)
    inside_r = (in_frac > 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(np.pad(inside_r, 1), cv2.DIST_L2, 3)[1:-1, 1:-1] * f
    d_out = cv2.distanceTransform((1 - inside_r).astype(np.uint8), cv2.DIST_L2, 3) * f
    orig = out_frac > 0.5
    ring_r = (d_out > inset) & (d_out <= inset + ring) & orig
    inner_r = d_in > inset
    if not inner_r.any():
        inner_r = inside_r > 0
    if not ring_r.any() or not inner_r.any():
        return TextureInfo(False, reason="אין סביבה למדידה")

    # ---- 2. מגן: יחס טקסטורה כללי (כמו quality.texture_ratio)
    rin2 = float(s_in[inner_r].sum() / max(c_in[inner_r].sum(), 1.0))
    rout2 = float(s_out[ring_r].sum() / max(c_out[ring_r].sum(), 1.0))
    ratio = (math.sqrt(rin2) + _EPS_RMS) / (math.sqrt(rout2) + _EPS_RMS)
    if ratio >= guard:
        return TextureInfo(False, ratio, reason="הטקסטורה כבר דומה לסביבה")

    # ---- 3. יעד מקומי וחוסר
    sig = max(1.0, max(2 * ring, 24) / f)
    wide = (d_out > inset) & (d_out <= inset + 2 * ring) & orig
    wt = (out_frac * wide).astype(np.float32)
    c = 0.02
    e_t = (_gauss(mean_out * wt, sig) + c * rout2) / (_gauss(wt, sig) + c)
    w_i = (in_frac * inside_r).astype(np.float32)
    e_i = _gauss(mean_in * w_i, sig) / np.maximum(_gauss(w_i, sig), 1e-6)
    amp_r = (np.sqrt(np.maximum(e_t - e_i, 0.0)) * float(strength)).astype(np.float32)
    amp_rms = float(math.sqrt(float((amp_r.astype(np.float64) ** 2 * w_i).sum() / max(float(w_i.sum()), 1e-9))))
    if amp_rms < _MIN_AMP:
        return TextureInfo(False, ratio, amp_rms, "אין חוסר טקסטורה")

    # ---- 4. הוספת הגרעין, בפסים, רק בפיקסלי הקבוצה
    norm = _grain_norm()
    bw = x1 - x0
    base_row = y0 - cy0
    for k, r0 in enumerate(range(base_row, y1 - cy0, STRIP)):
        cancel.raise_if_cancelled()
        r1 = min(y1 - cy0, r0 + STRIP)
        sel = sel_g[r0 - base_row:r1 - base_row]
        if not sel.any():
            continue
        q0, q1 = max(0, r0 // f - 1), min(rh, -(-r1 // f) + 1)
        up = cv2.resize(amp_r[q0:q1], (rw * f, (q1 - q0) * f), interpolation=cv2.INTER_LINEAR)
        amp = up[r0 - q0 * f:r1 - q0 * f, x0 - cx0:x1 - cx0]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, x0, y0, x1, y1, k]))
        n = rng.standard_normal((r1 - r0 + 2 * HALO, bw + 2 * HALO), dtype=np.float32)
        delta = _grain(n)[HALO:-HALO, HALO:-HALO] * (amp / norm)
        dst = result[y0 + r0 - base_row:y0 + r1 - base_row, x0:x1]        # view של result
        # הגרעין מעוגל לשלם לפני ההוספה: זהה בדיוק בשלושת הערוצים (חיבור float32 היה מעגל אחרת בעוצמות שונות)
        out = np.clip(dst.astype(np.int16) + np.rint(delta).astype(np.int16)[..., None], 0, 255).astype(np.uint8)
        dst[sel] = out[sel]
    return TextureInfo(True, ratio, amp_rms)
