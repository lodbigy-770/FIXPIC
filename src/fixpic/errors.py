"""שגיאות מוגדרות של המנוע (ARCHITECTURE §10). כל חריגה פנימית מומרת לאחת מאלה."""


class FixPicError(Exception):
    """בסיס לכל שגיאות המנוע."""


class ModelLoadError(FixPicError):
    """קובץ מודל חסר, LFS pointer, hash לא תואם, או כשל ביצירת Session."""


class InferenceError(FixPicError):
    """כשל בהרצת המודל (חריגה ב-backend, OOM)."""


class EmptyMaskError(FixPicError):
    """לא סומן אזור: אין מה להריץ."""


class InvalidInputError(FixPicError):
    """תמונה/מסכה בפורמט, טיפוס או גודל לא תקינים."""


class CancelledError(FixPicError):
    """הפעולה בוטלה על ידי המשתמש; התמונה לא משתנה."""


class OutOfMemoryError(FixPicError):
    """הקצאת זיכרון נכשלה (בהרצת המודל או בעיבוד התמונה), או שההערכה חורגת מהתקרה שהוגדרה.
    אינו InferenceError בכוונה: זה אינו כשל backend, ולכן אין מעבר ל-fallback."""


def is_oom(exc: BaseException) -> bool:
    """זיהוי כשל זיכרון מחריגות של numpy/OpenVINO/ORT (אין להן טיפוס אחיד)."""
    if isinstance(exc, (MemoryError, OutOfMemoryError)):
        return True
    t = str(exc).lower()
    return any(k in t for k in ("bad_alloc", "failed to allocate", "out of memory", "allocate memory",
                                "cannot allocate", "unable to allocate"))
