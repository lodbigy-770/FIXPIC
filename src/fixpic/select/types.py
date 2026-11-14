"""טיפוסי בחירה אוטומטית (AUTO-SELECT.md §5): קלט (`Prompts`), הצעה (`Proposal`), `Context` ושגיאה.

כל הקואורדינטות בפיקסלי התמונה המלאה. `Proposal` באותו פורמט ש-`mask_ops.polygon_region` מחזירה (חלון + מלבן), כך
שהמסכה לעולם אינה נבנית בגודל התמונה, ומאושרת ל-`MaskLayer` דרך `add_window`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ..errors import FixPicError

Rect = tuple[int, int, int, int]   # x0, y0, x1, y1 (x1/y1 בלעדיים), כמו ב-MaskLayer


class SelectionError(FixPicError):
    """קלט בחירה לא תקין (נקודה מחוץ לתמונה, חסרה נקודה, קלט שה-Provider לא תומך בו), או כשל ב-Provider.
    הצעה ריקה אינה שגיאה: `propose` מחזיר None."""


class Point(NamedTuple):
    x: float
    y: float
    positive: bool = True       # False = נקודה שלילית ("לא זה")


@dataclass(frozen=True)
class Prompts:
    """קלט המשתמש. שדה שה-Provider אינו תומך בו = `SelectionError` (לא מתעלמים בשקט)."""
    points: tuple[Point, ...] = ()
    box: Rect | None = None                                    # מלבן גס סביב האובייקט (או חיתוך אזור החיפוש)
    polygon: tuple[tuple[float, float], ...] | None = None     # לאסו גס
    tolerance: int | None = None                               # שרביט: 0..255, None = ברירת מחדל של ה-Provider

    @property
    def positives(self) -> tuple[Point, ...]:
        return tuple(p for p in self.points if p.positive)

    @property
    def negatives(self) -> tuple[Point, ...]:
        return tuple(p for p in self.points if not p.positive)


@dataclass(frozen=True)
class Context:
    """תוצר `prepare`, אטום ללקוח: מה שה-Provider צריך כדי לענות מהר. `shape` = (h, w) של התמונה המלאה."""
    provider: str
    shape: tuple[int, int]
    data: Any = None
    nbytes: int = 0            # להערכת זיכרון (תקרת ה-cache)


@dataclass(frozen=True)
class Proposal:
    """הצעת בחירה: `window` (uint8, 0/255) בגודל `rect` בתוך התמונה. `score` 0..1. `alternatives` = הצעות מועמדות
    נוספות (רמה 1: חלק/אובייקט/שלם); ריק ברמה 0. ההצעה אינה נוגעת ב-`MaskLayer` עד `add_to`."""
    window: np.ndarray
    rect: Rect
    score: float = 1.0
    alternatives: tuple["Proposal", ...] = ()

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.rect
        w = self.window
        if not (isinstance(w, np.ndarray) and w.dtype == np.uint8 and w.ndim == 2):
            raise SelectionError("חלון ההצעה חייב להיות uint8 [H,W]")
        if not (x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0) or w.shape != (y1 - y0, x1 - x0):
            raise SelectionError("מלבן ההצעה אינו תואם את החלון")

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.window))

    def add_to(self, layer, *, erase: bool = False) -> Rect | None:
        """מאשר את ההצעה לשכבת מסכה (`MaskLayer.add_window`); עם `erase=True` מחסיר אותה. כדאי לעטוף ב-
        `MaskHistory.do` כדי שיהיה צעד Undo אחד. מחזיר את המלבן ששונה, או None אם לא השתנה דבר."""
        return layer.add_window(self.rect, self.window, erase=erase)
