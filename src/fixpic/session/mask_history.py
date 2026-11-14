"""MaskHistory (M4, ללא UI): Undo/Redo לעריכות של `MaskLayer` (ציור, מחיקה, לאסו, הרחבה/כיווץ, מחיקת רכיב...).

נפרד מ-`History` של התמונה (שמחזיק Patches של Commit): המסכה היא מצב עבודה שלפני שליחה ל-Job, ועריכותיה לא
נוגעות בתמונה. ה-UI מחליט איזו היסטוריה מקבלת Ctrl+Z (למשל: קודם עריכות מסכה שטרם נשלחו, ואז Commit-ים).

צעד = חלון דק סביב השינוי + הפרש XOR בינארי (המסכה 0/255, לכן אותו הפרש מבטל וגם מבצע מחדש) + `brush_px` לפני
ואחרי. הזיכרון לפי גודל השינוי ולא לפי התמונה. מגבלות: מספר צעדים וגם זיכרון; תמיד נשמר לפחות הצעד האחרון.

שימוש:
  `h.do(lambda m: m.grow(3))`            פעולה בודדת; מחזיר את תוצאת הפעולה
  `h.begin()` ... ציור ... `h.end()`    משיכה שלמה (כמה `paint`) כצעד אחד; `h.end(cancel=True)` מחזיר את המצב
שינוי שבוצע על השכבה מחוץ ל-`do`/`begin..end` (למשל `Session.submit_stroke` שמרוקן אותה, או `add` אחרי כשל)
הופך את ההיסטוריה ל"מיושנת" (`stale`): ההפרשים כבר לא תואמים. הפעולה הבאה מנקה אותה, ו-`undo`/`redo` מחזירים None.
לא בטוחה לשימוש מכמה threads (כמו `MaskLayer`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import cv2
import numpy as np

from ..errors import InvalidInputError
from .mask_layer import MaskLayer, Rect

T = TypeVar("T")

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_BYTES = 64 * 2**20


@dataclass(frozen=True)
class MaskStep:
    rect: Rect
    diff: np.ndarray          # uint8 0/255, בגודל החלון
    brush_before: float | None
    brush_after: float | None

    @property
    def nbytes(self) -> int:
        return int(self.diff.nbytes)


class MaskHistory:
    def __init__(self, layer: MaskLayer, max_steps: int = DEFAULT_MAX_STEPS, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_steps < 1 or max_bytes < 1:
            raise ValueError("max_steps ו-max_bytes חייבים להיות חיוביים")
        self.layer = layer
        self.max_steps, self.max_bytes = max_steps, max_bytes
        self._undo: list[MaskStep] = []
        self._redo: list[MaskStep] = []
        self._version = layer.version     # גרסת השכבה כפי שההיסטוריה מכירה אותה
        self._open: tuple[np.ndarray, float | None] | None = None   # (snapshot, brush_px) של begin()
        self.dropped_total = 0

    # ------------------------------------------------------------------ מצב
    @property
    def stale(self) -> bool:
        """השכבה השתנתה מחוץ להיסטוריה; הצעדים השמורים כבר לא תואמים אותה."""
        return self._open is None and self.layer.version != self._version

    @property
    def can_undo(self) -> bool:
        return not self.stale and bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return not self.stale and bool(self._redo)

    @property
    def undo_count(self) -> int:
        return 0 if self.stale else len(self._undo)

    @property
    def redo_count(self) -> int:
        return 0 if self.stale else len(self._redo)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._undo) + sum(s.nbytes for s in self._redo)

    @property
    def recording(self) -> bool:
        return self._open is not None

    def clear(self) -> None:
        """מנקה את הצעדים ומסנכרן לגרסה הנוכחית של השכבה (בטוח לקרוא גם באמצע `begin`: מבטל אותו)."""
        self._undo.clear()
        self._redo.clear()
        self._open = None
        self._version = self.layer.version

    # ------------------------------------------------------------- הקלטה
    def begin(self) -> None:
        """פותח צעד: כל מה שייעשה בשכבה עד `end` הוא צעד אחד. `InvalidInputError` אם כבר פתוח."""
        if self._open is not None:
            raise InvalidInputError("begin כבר נקרא")
        self._drop_if_stale()
        self._open = (self.layer.snapshot(), self.layer.brush_px)

    def end(self, *, cancel: bool = False) -> bool:
        """סוגר את הצעד. `cancel=True` מחזיר את השכבה למצב שב-`begin` ולא שומר דבר. מחזיר True אם נשמר צעד
        (False: לא השתנה דבר, או בוטל)."""
        if self._open is None:
            raise InvalidInputError("end ללא begin")
        before, brush_before = self._open
        self._open = None
        if cancel:
            if self.layer.version != self._version:
                self.layer.restore(before)
                self.layer.apply_diff((0, 0, 1, 1), np.zeros((1, 1), np.uint8), brush_before)   # שחזור brush_px
            self._version = self.layer.version
            return False
        if self.layer.version == self._version:
            return False
        diff = np.bitwise_xor(before, self.layer.array)
        x, y, w, h = cv2.boundingRect(diff)
        if w == 0 or h == 0:        # השתנה רק brush_px או שינוי שאינו משנה פיקסלים: אין מה לבטל
            self._version = self.layer.version
            return False
        step = MaskStep((x, y, x + w, y + h), np.ascontiguousarray(diff[y:y + h, x:x + w]),
                        brush_before, self.layer.brush_px)
        self._redo.clear()
        self._undo.append(step)
        total = sum(s.nbytes for s in self._undo)
        while len(self._undo) > 1 and (len(self._undo) > self.max_steps or total > self.max_bytes):
            total -= self._undo.pop(0).nbytes
            self.dropped_total += 1
        self._version = self.layer.version
        return True

    def do(self, op: Callable[[MaskLayer], T]) -> T:
        """מריץ `op(layer)` כצעד אחד. אם `op` זורקת, השכבה חוזרת למצבה והחריגה ממשיכה."""
        self.begin()
        try:
            res = op(self.layer)
        except BaseException:
            self.end(cancel=True)
            raise
        self.end()
        return res

    # ------------------------------------------------------------- Undo/Redo
    def undo(self) -> Rect | None:
        """מבטל את הצעד האחרון; מחזיר את המלבן ששונה (לציור מחדש), או None אם אין מה לבטל/ההיסטוריה מיושנת."""
        if self._open is not None:
            raise InvalidInputError("יש צעד פתוח (begin)")
        if not self.can_undo:
            return None
        step = self._undo.pop()
        self.layer.apply_diff(step.rect, step.diff, step.brush_before)
        self._redo.append(step)
        self._version = self.layer.version
        return step.rect

    def redo(self) -> Rect | None:
        if self._open is not None:
            raise InvalidInputError("יש צעד פתוח (begin)")
        if not self.can_redo:
            return None
        step = self._redo.pop()
        self.layer.apply_diff(step.rect, step.diff, step.brush_after)
        self._undo.append(step)
        self._version = self.layer.version
        return step.rect

    # ---------------------------------------------------------------- פנימי
    def _drop_if_stale(self) -> None:
        if self.layer.version != self._version:
            self.clear()
