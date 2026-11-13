"""טעינה ושמירה של תמונות (M3.1; ARCHITECTURE §5, PARITY-SPEC §2 "ייבוא בכל גודל" ו-"הורדה/ייצוא").

חוזים:
- תמונה פנימית: RGB uint8 [H,W,3], C-contiguous וניתנת לכתיבה (כמו שה-Pipeline דורש). Alpha נשמר בנפרד כ-uint8
  [H,W] (או None). Alpha אטום לגמרי נזרק (אין שקיפות לשמר, וחוסך ערוץ בכל עיבוד).
- נתיבי Unicode: הקובץ נקרא ונכתב דרך `Path.read_bytes` / כתיבה בינארית, ופענוח/קידוד בזיכרון (לא `cv2.imread`).
- EXIF Orientation מיושם בטעינה, והתג מאופס ב-`exif` המוחזר, כך ששמירה לא מסובבת פעם שנייה.
- ICC נשמר רק כשהוא תואם ל-RGB (לא לגווני אפור/CMYK/16 סיביות שהומרו).
- שמירה אטומית: קידוד לזיכרון, כתיבה לקובץ זמני באותה תיקייה, ואז `os.replace`. כשל לא משאיר קובץ חלקי ולא
  פוגע בקובץ קיים.
- תמונה מונפשת (GIF/WebP): נטען הפריים הראשון בלבד (`LoadedImage.animated` = True).
- שגיאות: תוכן פגום/פורמט לא נתמך/גדול מדי = `InvalidInputError`; כשל מערכת קבצים = `ImageIOError`."""
from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..errors import ImageIOError, InvalidInputError

MAX_PIXELS = 150_000_000          # מעל זה: נדחה (תמונה 24MP היא כ-72MB; 150MP כ-450MB לתמונה אחת)
_LOAD_FORMATS = {"PNG", "JPEG", "MPO", "WEBP", "BMP", "TIFF", "GIF"}
SAVE_EXTENSIONS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
_MAX_DIM = {"WEBP": 16383, "JPEG": 65500}
_ORIENTATION_TAG = 0x0112
_RGB_ICC_MODES = {"RGB", "RGBA", "RGBX", "P", "PA", "RGBa"}   # מצבים שבהם פרופיל ה-ICC הוא של RGB


@dataclass
class LoadedImage:
    rgb: np.ndarray                   # uint8 [H,W,3]
    alpha: np.ndarray | None = None   # uint8 [H,W], None = אטום
    icc_profile: bytes | None = None
    exif: bytes | None = None         # Orientation כבר אופס
    source_format: str = ""           # "PNG" / "JPEG" / "WEBP" ...
    source_path: Path | None = None
    animated: bool = False            # נטען הפריים הראשון בלבד
    orientation_applied: int = 1      # ערך ה-EXIF Orientation שיושם (1 = ללא)
    depth_reduced: bool = False       # היה מעל 8 סיביות לערוץ והומר ל-8

    @property
    def size(self) -> tuple[int, int]:
        """(רוחב, גובה)."""
        return self.rgb.shape[1], self.rgb.shape[0]


@dataclass
class SaveResult:
    path: Path
    format: str
    bytes_written: int
    alpha_flattened: bool = False     # ה-Alpha הונח על רקע לבן כי הפורמט אינו תומך בשקיפות


# ---------------------------------------------------------------------------------------------- טעינה
def load_image(path: str | os.PathLike) -> LoadedImage:
    """קריאת קובץ. `ImageIOError` אם אי אפשר לקרוא אותו; `InvalidInputError` אם התוכן אינו תמונה נתמכת."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise ImageIOError(f"לא ניתן לקרוא את הקובץ: {p} ({e})") from e
    return decode_image(data, source_path=p)


def decode_image(data: bytes, *, source_path: Path | None = None) -> LoadedImage:
    """פענוח מבייטים (קובץ שנקרא, גרירה, הדבקה). קובץ פגום או לא נתמך = InvalidInputError."""
    if not data:
        raise InvalidInputError("הקובץ ריק")
    try:
        return _decode(data, source_path)
    except (InvalidInputError, MemoryError):
        raise
    except Image.DecompressionBombError as e:
        raise InvalidInputError(f"התמונה גדולה מדי ({e})") from e
    except Exception as e:   # Pillow זורק OSError/ValueError/SyntaxError/EOFError לפי הפורמט והנזק
        raise InvalidInputError(f"הקובץ פגום או אינו תמונה נתמכת ({type(e).__name__}: {e})") from e


def _decode(data: bytes, source_path: Path | None) -> LoadedImage:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)   # המגבלה שלנו (MAX_PIXELS) היא הקובעת
        img = Image.open(io.BytesIO(data))
        fmt = img.format or ""
        if fmt not in _LOAD_FORMATS:
            raise InvalidInputError(f"פורמט לא נתמך: {fmt or 'לא מזוהה'}")
        w, h = img.size
        if w < 1 or h < 1:
            raise InvalidInputError("תמונה ריקה")
        if w * h > MAX_PIXELS:
            raise InvalidInputError(f"התמונה גדולה מדי: {w}x{h} ({w * h / 1e6:.0f}MP, המגבלה {MAX_PIXELS // 10**6}MP)")
        animated = getattr(img, "n_frames", 1) > 1
        orig_mode = img.mode
        icc = img.info.get("icc_profile") if orig_mode in _RGB_ICC_MODES else None
        img.load()
        orientation = int(img.getexif().get(_ORIENTATION_TAG, 1) or 1)
        if orientation != 1:
            img = ImageOps.exif_transpose(img)
        exif = _clean_exif(img)
        rgb, alpha, reduced = _to_arrays(img)
    return LoadedImage(rgb, alpha, icc, exif, "JPEG" if fmt == "MPO" else fmt, source_path, animated,
                       orientation, reduced)


def _clean_exif(img: Image.Image) -> bytes | None:
    """EXIF כבייטים עם Orientation מאופס; None אם אין או שאי אפשר לסדר אותו (נתון משני, לא מכשיל טעינה)."""
    try:
        ex = img.getexif()
        if _ORIENTATION_TAG in ex:
            del ex[_ORIENTATION_TAG]
        return ex.tobytes() if len(ex) else None
    except Exception:
        return None


def _to_arrays(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None, bool]:
    mode, reduced = img.mode, False
    if mode.startswith("I"):            # 16/32 סיביות באפור: מיפוי עם עיגול (65535 -> 255)
        a = np.asarray(img).astype(np.int64)
        a = np.clip(a, 0, 65535)
        g = ((a * 255 + 32767) // 65535).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(g[..., None], 3, axis=2)), None, True
    has_alpha = mode in ("RGBA", "LA", "PA", "RGBa") or (mode == "P" and "transparency" in img.info)
    if has_alpha:
        a = np.array(img.convert("RGBA"), dtype=np.uint8)
        alpha = np.ascontiguousarray(a[..., 3])
        rgb = np.ascontiguousarray(a[..., :3])
        del a
        return rgb, (None if bool((alpha == 255).all()) else alpha), reduced
    if mode != "RGB":
        img = img.convert("RGB")        # L, 1, P, CMYK, YCbCr, LAB, HSV, F ...
    return np.array(img, dtype=np.uint8), None, reduced


def image_from_array(arr: np.ndarray, *, source_format: str = "") -> LoadedImage:
    """בניית תמונה מ-ndarray (הדבקה מהלוח דרך QImage): uint8 [H,W], [H,W,1], [H,W,3] או [H,W,4] (RGB/RGBA)."""
    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8 or arr.ndim not in (2, 3) or arr.size == 0:
        raise InvalidInputError("נדרש מערך uint8 בצורה [H,W], [H,W,1], [H,W,3] או [H,W,4]")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise InvalidInputError(f"מספר ערוצים לא נתמך: {arr.shape[2]}")
    h, w = arr.shape[:2]
    if h * w > MAX_PIXELS:
        raise InvalidInputError(f"התמונה גדולה מדי: {w}x{h}")
    if arr.ndim == 2 or arr.shape[2] == 1:
        g = arr.reshape(h, w)
        return LoadedImage(np.ascontiguousarray(np.repeat(g[..., None], 3, axis=2)), source_format=source_format)
    rgb = np.array(arr[..., :3], dtype=np.uint8, order="C")
    alpha = None
    if arr.shape[2] == 4 and not bool((arr[..., 3] == 255).all()):
        alpha = np.array(arr[..., 3], dtype=np.uint8, order="C")
    return LoadedImage(rgb, alpha, source_format=source_format)


# ---------------------------------------------------------------------------------------------- שמירה
def composite_on(rgb: np.ndarray, alpha: np.ndarray | None, bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """הנחת תמונה עם Alpha על רקע אחיד (לפורמט בלי שקיפות). בלי Alpha: מחזיר את אותה תמונה."""
    if alpha is None:
        return rgb
    a = alpha.astype(np.uint16)[..., None]
    out = (rgb.astype(np.uint16) * a + np.array(bg, np.uint16) * (255 - a) + 127) // 255
    return out.astype(np.uint8)


def encode_image(rgb: np.ndarray, alpha: np.ndarray | None = None, *, fmt: str, quality: int = 92,
                 lossless: bool = False, icc_profile: bytes | None = None, exif: bytes | None = None) -> tuple[bytes, bool]:
    """קידוד לזיכרון. fmt: "PNG" / "JPEG" / "WEBP". מחזיר (בייטים, האם ה-Alpha הונח על רקע לבן)."""
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in ("PNG", "JPEG", "WEBP"):
        raise InvalidInputError(f"פורמט שמירה לא נתמך: {fmt}")
    _check_arrays(rgb, alpha)
    if not 1 <= int(quality) <= 100:
        raise InvalidInputError("איכות חייבת להיות בין 1 ל-100")
    h, w = rgb.shape[:2]
    if max(h, w) > _MAX_DIM.get(fmt, 10**9):
        raise InvalidInputError(f"{fmt} מוגבל ל-{_MAX_DIM[fmt]} פיקסלים בצלע; התמונה {w}x{h}. שמרו כ-PNG")
    flattened = alpha is not None and fmt == "JPEG"
    if flattened:
        rgb, alpha = composite_on(rgb, alpha), None
    arr = rgb if alpha is None else np.dstack((rgb, alpha))
    img = Image.fromarray(arr)
    kw: dict = {}
    if icc_profile:
        kw["icc_profile"] = icc_profile
    if exif:
        kw["exif"] = exif
    if fmt == "PNG":
        kw["compress_level"] = 6
    elif fmt == "JPEG":
        kw.update(quality=int(quality), optimize=True)
    else:
        kw.update(quality=int(quality), lossless=bool(lossless), method=4)
    buf = io.BytesIO()
    try:
        img.save(buf, fmt, **kw)
    except MemoryError:
        raise
    except Exception as e:
        raise InvalidInputError(f"קידוד {fmt} נכשל ({type(e).__name__}: {e})") from e
    return buf.getvalue(), flattened


def save_image(path: str | os.PathLike, rgb: np.ndarray, alpha: np.ndarray | None = None, *, fmt: str | None = None,
               quality: int = 92, lossless: bool = False, icc_profile: bytes | None = None,
               exif: bytes | None = None) -> SaveResult:
    """שמירה אטומית ברזולוציה מלאה. הפורמט נקבע לפי הסיומת (.png/.jpg/.jpeg/.webp) אלא אם נמסר `fmt`.
    כשל קידוד = InvalidInputError; כשל כתיבה = ImageIOError. בשני המקרים קובץ קיים לא נפגע."""
    p = Path(path)
    if fmt is None:
        fmt = SAVE_EXTENSIONS.get(p.suffix.lower())
        if fmt is None:
            raise InvalidInputError(f"סיומת לא נתמכת לשמירה: '{p.suffix}' (נתמך: PNG, JPG, WebP)")
    data, flattened = encode_image(rgb, alpha, fmt=fmt, quality=quality, lossless=lossless,
                                   icc_profile=icc_profile, exif=exif)
    if not p.parent.is_dir():
        raise ImageIOError(f"התיקייה לא קיימת: {p.parent}")
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ImageIOError(f"השמירה נכשלה: {p} ({e})") from e
    return SaveResult(p, "JPEG" if fmt.upper() == "JPG" else fmt.upper(), len(data), flattened)


def _check_arrays(rgb: np.ndarray, alpha: np.ndarray | None) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise InvalidInputError("נדרשת תמונה uint8 בצורה [H,W,3] (RGB)")
    if alpha is not None and (not isinstance(alpha, np.ndarray) or alpha.dtype != np.uint8
                              or alpha.shape != rgb.shape[:2]):
        raise InvalidInputError("Alpha חייב להיות uint8 בגודל התמונה [H,W]")
