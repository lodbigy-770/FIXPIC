"""חוזה `SelectionProvider` (AUTO-SELECT.md §5): כללי מספיק לרמה 0 (שרביט, GrabCut) ולרמה 1 (מודל)."""
from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from ..models.base import CancelToken
from .types import Context, Prompts, Proposal


@runtime_checkable
class SelectionProvider(Protocol):
    """- `prepare(rgb, cancel)`: עבודה כבדה פעם אחת לגרסת תמונה (למודל: Embedding). `rgb` uint8 [H,W,3], אינו משתנה.
    - `propose(ctx, prompts, cancel)`: מהיר. `Proposal`, או None אם הבחירה ריקה. `SelectionError` על קלט לא תקין או
      קלט שלא נתמך. ביטול נבדק בין שלבים (`CancelledError`).
    - `heavy`: True = `prepare` ו-`propose` מתחרים ב-Inpaint על ה-CPU, ולכן ה-Service ימתין כש-Inpaint פעיל.
    - `stale_ok`: True = Context של גרסת תמונה קודמת עדיין שימושי (מודל Embedding: הסרת אובייקט אחר כמעט אינה משפיעה).
      False = חייבים Context מדויק (שרביט: עובד על הצבעים עצמם). **[לא נבדק עם מודל]**."""
    name: str
    heavy: ClassVar[bool]
    stale_ok: ClassVar[bool]

    def prepare(self, rgb: np.ndarray, cancel: CancelToken | None = None) -> Context: ...

    def propose(self, ctx: Context, prompts: Prompts, cancel: CancelToken | None = None) -> Proposal | None: ...
