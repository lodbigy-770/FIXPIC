"""History (ARCHITECTURE §9): שני מחסנים (Undo/Redo) של Patches, מוגבלים במספר צעדים *וגם* בזיכרון.

- Redo מתאפס בכל Push חדש.
- גלישה מוחקת את הישנים ביותר. תמיד נשמר לפחות הצעד האחרון, גם אם לבדו חורג מתקרת הזיכרון
  (אחרת Undo של הפעולה האחרונה לא יהיה אפשרי בדיוק כשהיא הכי גדולה).
- ל-History אין ידיעה על התמונה: `Session` מפעיל את ה-Patches ומעדכן את המחסנים אחרי הצלחה בלבד.
- `current_id`: מזהה ה-Patch שבראש מחסן ה-Undo (0 = מצב התחלתי). מזהים עולים ואינם ממוחזרים, ולכן שוויון
  בין שני ערכים אומר "אותו מצב" (בסיס לחיווי "שונה מאז השמירה")."""
from __future__ import annotations

import itertools

from .patch import Patch

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_BYTES = 256 * 2**20


class History:
    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_steps < 1 or max_bytes < 1:
            raise ValueError("max_steps ו-max_bytes חייבים להיות חיוביים")
        self.max_steps, self.max_bytes = max_steps, max_bytes
        self._undo: list[Patch] = []
        self._redo: list[Patch] = []
        self._ids = itertools.count(1)
        self.dropped_total = 0          # כמה צעדים נמחקו בגלל המגבלות (ל-UI: "ההיסטוריה נחתכה")

    def next_id(self) -> int:
        return next(self._ids)

    # ------------------------------------------------------------------ מצב
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_count(self) -> int:
        return len(self._undo)

    @property
    def redo_count(self) -> int:
        return len(self._redo)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self._undo) + sum(p.nbytes for p in self._redo)

    @property
    def current_id(self) -> int:
        return self._undo[-1].id if self._undo else 0

    def peek_undo(self) -> Patch | None:
        return self._undo[-1] if self._undo else None

    def peek_redo(self) -> Patch | None:
        return self._redo[-1] if self._redo else None

    # ---------------------------------------------------------------- שינוי
    def push(self, patch: Patch) -> int:
        """מוסיף Patch חדש, מאפס Redo וחותך ישנים לפי המגבלות. מחזיר כמה צעדים נמחקו."""
        self._redo.clear()
        self._undo.append(patch)
        dropped = 0
        total = sum(p.nbytes for p in self._undo)
        while len(self._undo) > 1 and (len(self._undo) > self.max_steps or total > self.max_bytes):
            total -= self._undo.pop(0).nbytes
            dropped += 1
        self.dropped_total += dropped
        return dropped

    def mark_undone(self) -> None:
        """מעביר את ראש ה-Undo ל-Redo (אחרי ש-Session החיל אותו בהצלחה)."""
        self._redo.append(self._undo.pop())

    def mark_redone(self) -> None:
        self._undo.append(self._redo.pop())

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
