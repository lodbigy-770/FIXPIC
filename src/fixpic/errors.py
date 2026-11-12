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
