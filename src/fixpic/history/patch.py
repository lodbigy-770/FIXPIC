"""Patch (ARCHITECTURE §9, ADR-004): שינוי אחד בתמונה העובדת, מספיק כדי לבצע Undo ו-Redo בלי לשמור תמונה מלאה.

נשמרים רק הפיקסלים שבתוך המסכה (לא כל המלבן), לפני ואחרי, ללא דחיסה (6 בייט לפיקסל מסכה, ו-8 עם Alpha). רק
המסכה עצמה נדחסת. zlib על הפיקסלים נבדק ונדחה: על נתוני צילום היחס כ-0.65 בלבד, ובמחיר כ-0.3-0.5 שניות ל-4MP
בכל Commit (ב-thread של ה-UI).
`apply` לעולם אינו משנה את הקלט: מחזיר מערכים חדשים (Jobs ותצוגה עשויים להחזיק את המערך הקודם)."""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from ..errors import InvalidInputError

Box = tuple[int, int, int, int]   # x0, y0, x1, y1 (x1/y1 בלעדיים)
_LEVEL = 1   # דחיסת המסכה בלבד


def _own(a: np.ndarray) -> np.ndarray:
    """לוקח בעלות (ללא העתקה) וסוגר לכתיבה; המתקשר מעביר מערכים טריים (אינדוקס בוליאני כבר מעתיק)."""
    a = np.ascontiguousarray(a, dtype=np.uint8)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Patch:
    id: int
    box: Box
    image_hw: tuple[int, int]            # גודל התמונה שעליה נוצר; apply על גודל אחר נדחה
    count: int                           # מספר הפיקסלים במסכה
    mask: bytes                          # packbits של מסכת המלבן, דחוס
    before_rgb: np.ndarray               # [N,3] uint8, קריאה בלבד
    after_rgb: np.ndarray
    before_alpha: np.ndarray | None = None   # [N] uint8
    after_alpha: np.ndarray | None = None

    @classmethod
    def create(cls, patch_id: int, box: Box, image_hw: tuple[int, int], sub_mask: np.ndarray,
               before_rgb: np.ndarray, after_rgb: np.ndarray,
               before_alpha: np.ndarray | None = None, after_alpha: np.ndarray | None = None) -> "Patch":
        """sub_mask: bool בגודל המלבן; before/after: הפיקסלים במסכה בסדר שורות ([N,3], ו-[N] ל-Alpha).
        ה-Patch לוקח בעלות על המערכים (לא מעתיק): יש להעביר מערכים שאיש אחר לא ישנה."""
        n = int(sub_mask.sum())
        x0, y0, x1, y1 = box
        if sub_mask.shape != (y1 - y0, x1 - x0) or before_rgb.shape != (n, 3) or after_rgb.shape != (n, 3):
            raise InvalidInputError("Patch: צורות לא עקביות")
        if (before_alpha is None) != (after_alpha is None):
            raise InvalidInputError("Patch: Alpha לפני ואחרי חייבים להינתן יחד")
        return cls(patch_id, box, image_hw, n, zlib.compress(np.packbits(sub_mask.ravel()).tobytes(), _LEVEL),
                   _own(before_rgb), _own(after_rgb),
                   None if before_alpha is None else _own(before_alpha),
                   None if after_alpha is None else _own(after_alpha))

    @property
    def nbytes(self) -> int:
        return len(self.mask) + sum(a.nbytes for a in (self.before_rgb, self.after_rgb, self.before_alpha,
                                                       self.after_alpha) if a is not None)

    @property
    def has_alpha(self) -> bool:
        return self.before_alpha is not None

    def _sub_mask(self) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        h, w = y1 - y0, x1 - x0
        bits = np.unpackbits(np.frombuffer(zlib.decompress(self.mask), np.uint8), count=h * w)
        return bits.astype(bool).reshape(h, w)

    def apply(self, rgb: np.ndarray, alpha: np.ndarray | None, *, forward: bool
              ) -> tuple[np.ndarray, np.ndarray | None]:
        """forward=True: מצב "אחרי" (Redo / Commit); False: מצב "לפני" (Undo). מחזיר (rgb, alpha) חדשים."""
        if rgb.shape[:2] != self.image_hw or (alpha is not None and alpha.shape != self.image_hw):
            raise InvalidInputError("Patch לא מתאים לגודל התמונה הנוכחית")
        if self.has_alpha != (alpha is not None):
            raise InvalidInputError("Patch לא מתאים: חוסר התאמה בנוכחות Alpha")
        x0, y0, x1, y1 = self.box
        sub = self._sub_mask()
        new_rgb = rgb.copy()
        new_rgb[y0:y1, x0:x1][sub] = self.after_rgb if forward else self.before_rgb
        new_alpha = None
        if alpha is not None:
            new_alpha = alpha.copy()
            new_alpha[y0:y1, x0:x1][sub] = self.after_alpha if forward else self.before_alpha
        return new_rgb, new_alpha
