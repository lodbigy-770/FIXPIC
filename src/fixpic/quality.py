"""M3.7.3: מדדי איכות למילוי באזורים גדולים, בלי תמונת אמת (אין צורך בתוצאה "נכונה" להשוואה).

שלושה סימפטומים ששמענו עליהם, ומדד לכל אחד:

* **צל מטושטש** - `texture_ratio`: RMS של תדר גבוה בתוך האזור המולא חלקי RMS בטבעת הסביבה. ערך קרוב ל-1 = טקסטורה
  דומה לסביבה; נמוך משמעותית (למשל מתחת ל-0.5) = מריחה. ערך גבוה מ-1 משמעותית = רעש/טקסטורה שאינה מהסביבה.
* **פסים/תפרים** - `stripe_score` (בלי לדעת איפה התפרים: ציון חריגות של פרופיל הגרדיאנט לפי עמודה/שורה בתוך האזור)
  ו-`seam_ratio` (כשידוע מיקום התפר, למשל גבול אריח: הגרדיאנט על הקו מול קווים סמוכים).
* **הפרש צבע בגבול** - `boundary_color_diff`: הפרש הצבע הממוצע בין הצד הפנימי לחיצוני של גבול המסכה.

כל המדדים עובדים על התיבה של המסכה (עם שוליים) ולא על כל התמונה, ולכן מהירים גם ב-24MP. הם אינדיקציה לבדיקה
אוטומטית (ולבחירת תצורות ב-`quality_bench`), ואינם מחליפים עין. ספי הפרשנות שבתיעוד הם נקודת התחלה מכוילת על
תמונות סינתטיות בלבד (`tests/test_m373.py`); כיול על תמונות אמיתיות הוא חלק מ-M3.7.4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .errors import EmptyMaskError, InvalidInputError

_EPS_RMS = 0.25  # רצפת רעש (ברמות אפור) כדי שאזורים שטוחים משני הצדדים לא ייתנו יחס שרירותי


@dataclass(frozen=True)
class QualityReport:
    texture_ratio: float
    texture_rms_inside: float
    texture_rms_ring: float
    stripe_score: float
    boundary_color_diff: float
    boundary_color_diff_p95: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------- עזר

def _check(result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if result.ndim != 3 or result.shape[2] != 3 or result.dtype != np.uint8:
        raise InvalidInputError("התמונה חייבת להיות RGB uint8")
    if mask.shape != result.shape[:2]:
        raise InvalidInputError("מידות המסכה והתמונה שונות")
    m = mask > 0
    if not m.any():
        raise EmptyMaskError("המסכה ריקה")
    return m


def _auto_ring(m: np.ndarray) -> int:
    area = int(m.sum())
    return int(np.clip(round(0.1 * np.sqrt(area)), 8, 96))


def _crop(m: np.ndarray, pad: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return (slice(max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)),
            slice(max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)))


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _highpass(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


# ---------------------------------------------------------------- צל מטושטש

def texture_ratio(result: np.ndarray, mask: np.ndarray, *, ring_px: int | None = None) -> tuple[float, float, float]:
    """(יחס, RMS פנימי, RMS בטבעת). הפנים מוקטן ב-`inset` (בלי רצועת המיזוג), והטבעת רחוקה ממנו `inset` פיקסלים,
    רוחבה `ring_px` (ברירת מחדל: כ-10% מצלע שקולה של האזור, 8..96). הטבעת לא כוללת פיקסלים שבתוך המסכה."""
    m = _check(result, mask)
    ring = ring_px if ring_px is not None else _auto_ring(m)
    inset = max(2, ring // 4)
    ys, xs = _crop(m, ring + inset + 4)
    sub = m[ys, xs].astype(np.uint8)
    hp = _highpass(_luma(result[ys, xs]))
    # מרחקים (O(n)) במקום מורפולוגיה עם גרעין גדול: ב-6MP זה ההבדל בין עשיריות שנייה לכמה שניות
    d_in = cv2.distanceTransform(np.pad(sub, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    inner = d_in > inset
    if not inner.any():  # אזור צר מדי לשוליים: משתמשים במסכה עצמה
        inner = sub > 0
    d_out = cv2.distanceTransform((1 - sub).astype(np.uint8), cv2.DIST_L2, 3)
    ring_m = (d_out > inset) & (d_out <= inset + ring)
    if not ring_m.any():
        raise InvalidInputError("אין סביבה למדידה (המסכה ממלאת את כל התמונה)")
    rin = float(np.sqrt(np.mean(hp[inner] ** 2)))
    rout = float(np.sqrt(np.mean(hp[ring_m] ** 2)))
    return (rin + _EPS_RMS) / (rout + _EPS_RMS), rin, rout


# ---------------------------------------------------------------- פסים / תפרים

def _profile_z(prof: np.ndarray, valid: np.ndarray, win: int = 15) -> float:
    """ציון חריגות מקסימלי של פרופיל אחרי הסרת מגמה (חציון נע) ונרמול ב-MAD."""
    idx = np.nonzero(valid)[0]
    if idx.size < 2 * win:
        return 0.0
    p = prof[idx].astype(np.float32)
    k = win | 1
    pad = k // 2
    padded = np.pad(p, pad, mode="edge")
    trend = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    d = p - trend
    mad = float(np.median(np.abs(d - np.median(d))))
    scale = 1.4826 * mad + 0.02 * float(np.median(p)) + 1e-3
    return float(np.max(d) / scale)


def stripe_score(result: np.ndarray, mask: np.ndarray, *, min_support: float = 0.25) -> float:
    """מגלה תפרים/פסים בלי לדעת איפה הם: לכל עמודה (ושורה) בתוך האזור המולא (מוקטן קלות) הגרדיאנט האופקי
    (האנכי) הממוצע לאורך הפרופיל; הציון הוא החריגה העליונה הגבוהה ביותר מהמגמה, ביחידות סטיית תקן עמידה.
    אזור טבעי נותן בדרך כלל עד כ-4-5; תפר חד לאורך האזור נותן 8 ומעלה (כיול סינתטי בלבד). עמודות/שורות עם
    פחות מ-`min_support` מהתמיכה המרבית מדולגות, כדי שקצוות מסכה לא יופיעו כפסים."""
    m = _check(result, mask)
    ys, xs = _crop(m, 4)
    sub = (m[ys, xs]).astype(np.uint8)
    inner = cv2.erode(sub, _disk(3)) > 0
    if inner.sum() < 400:
        return 0.0
    g = _luma(result[ys, xs])
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    vx = inner[:, 1:] & inner[:, :-1]
    vy = inner[1:, :] & inner[:-1, :]
    cx = vx.sum(axis=0)
    cy = vy.sum(axis=1)
    px = (gx * vx).sum(axis=0) / np.maximum(cx, 1)
    py = (gy * vy).sum(axis=1) / np.maximum(cy, 1)
    zx = _profile_z(px, cx >= max(8, min_support * cx.max()))
    zy = _profile_z(py, cy >= max(8, min_support * cy.max()))
    return float(max(zx, zy))


def seam_ratio(result: np.ndarray, lines: list[tuple[str, int, int, int]], *, offset: int = 4, span: int = 3) -> float:
    """כשמיקום התפרים ידוע (למשל גבולות אריחים מ-`Pipeline.plan`): הגרדיאנט הממוצע לרוחב הקו חלקי הגרדיאנט הממוצע
    בקווים מקבילים במרחק `offset`..`offset+span-1` משני הצדדים. כל קו: `("v", x, y0, y1)` (קו אנכי בין עמודות x-1
    ל-x) או `("h", y, x0, x1)`. מחזיר את הממוצע על כל הקווים; 1 = אין תפר, 2 ומעלה = תפר בולט."""
    if not lines:
        raise InvalidInputError("לא ניתנו קווי תפר")
    g = _luma(result)
    h, w = g.shape
    out = []
    for kind, c, a, b in lines:
        if kind == "v":
            a, b = max(0, a), min(h, b)
            if a >= b or not (1 <= c < w):
                continue
            col = lambda x: np.abs(g[a:b, x] - g[a:b, x - 1]).mean() if 1 <= x < w else None  # noqa: E731
        elif kind == "h":
            a, b = max(0, a), min(w, b)
            if a >= b or not (1 <= c < h):
                continue
            col = lambda y: np.abs(g[y, a:b] - g[y - 1, a:b]).mean() if 1 <= y < h else None  # noqa: E731
        else:
            raise InvalidInputError(f"סוג קו לא מוכר: {kind}")
        near = [v for d in range(offset, offset + span) for v in (col(c - d), col(c + d)) if v is not None]
        if not near:
            continue
        out.append((col(c) + 0.5) / (float(np.mean(near)) + 0.5))
    if not out:
        raise InvalidInputError("כל קווי התפר מחוץ לתמונה")
    return float(np.mean(out))


# ---------------------------------------------------------------- הפרש צבע בגבול

def boundary_color_diff(result: np.ndarray, mask: np.ndarray, *, sigma: float | None = None) -> tuple[float, float]:
    """(ממוצע, אחוזון 95) של הפרש הצבע בין הממוצע המקומי בצד הפנימי לממוצע המקומי בצד החיצוני, בפיקסלי הגבול
    (רמות 0..255, ממוצע על ערוצים). הממוצעים המקומיים מנורמלים-לפי-צד (טשטוש גאוסי של הצד חלקי טשטוש המסכה שלו),
    כך שרקע לא נכנס לצד הפנימי ולהפך. `sigma` ברירת מחדל כ-3% מצלע שקולה (3..24)."""
    m = _check(result, mask)
    s = sigma if sigma is not None else float(np.clip(0.03 * np.sqrt(m.sum()), 3, 24))
    pad = int(4 * s) + 4
    ys, xs = _crop(m, pad)
    sub = m[ys, xs]
    img = result[ys, xs].astype(np.float32)
    f = max(1, int(s // 4))  # טשטוש רחב מבוצע על תמונה מוקטנת (הטשטוש חלק, כך שהתוצאה כמעט זהה)
    if f > 1:
        sw, sh = max(1, sub.shape[1] // f), max(1, sub.shape[0] // f)
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        sub = cv2.resize(sub.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
        s = s / f
    inside = sub.astype(np.float32)
    outside = 1.0 - inside
    edge = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    def local(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        den = cv2.GaussianBlur(weight, (0, 0), s)
        num = cv2.GaussianBlur(img * weight[..., None], (0, 0), s)
        return num / np.maximum(den, 1e-6)[..., None], den

    a_in, d_in = local(inside)
    a_out, d_out = local(outside)
    ok = edge & (d_in > 0.05) & (d_out > 0.05)
    if not ok.any():
        raise InvalidInputError("אין גבול למדידה (המסכה ממלאת את כל התמונה)")
    diff = np.abs(a_in - a_out).mean(axis=2)[ok]
    return float(diff.mean()), float(np.percentile(diff, 95))


# ---------------------------------------------------------------- מקבץ

def measure(result: np.ndarray, mask: np.ndarray, *, ring_px: int | None = None) -> QualityReport:
    """כל המדדים הבלתי-תלויים-במיקום-תפר, בדוח אחד. `result` = התמונה אחרי המילוי (RGB uint8, אותו גודל כמסכה)."""
    ratio, rin, rout = texture_ratio(result, mask, ring_px=ring_px)
    mean_d, p95_d = boundary_color_diff(result, mask)
    return QualityReport(ratio, rin, rout, stripe_score(result, mask), mean_d, p95_d)
