"""בקר כלי השרביט (M4, AUTO-SELECT שלב 4א): לחיצה → הצעה (בירוק) → אישור/ביטול. יורש מ-`SmartSelectController` (שלב 4ב).

משתמש בכל המנגנון של בקר הבחירה החכמה (שירות בחירה עם Worker יחיד, ביטול בקשה קודמת, מחוון אחרי 200ms, אישור כצעד אחד ב-`MaskHistory`,
גלגול אחורה בכשל, ניקוי אחרי שינוי תמונה) ורק מחליף את הספק ב-`MagicWandProvider` ומוסיף קלט נקודה וסבילות.

חוזים (בנוסף לאלה של `SmartSelectController`):
- **הקלט הוא נקודת זרע אחת** (`Prompts(points=(Point,), tolerance=...)`); לחיצה חדשה מחליפה אותה. אין משיכה מתקנת בשרביט.
- **גרירה אופקית משנה סבילות בתצוגה חיה:** `press` זוכר את הסבילות הנוכחית כבסיס, ו-`drag(dx)` קובע
  `בסיס + round(dx / DRAG_PX_PER_STEP)` (מוגבל ל-0..`MAX_UI_TOLERANCE`) ומבקש הצעה חדשה על אותה נקודה. אותו דבר ל-`set_tolerance`
  (סליידר). הסבילות נשמרת בין לחיצות (ושידור `toleranceChanged` מסנכרן את הלוח).
- **Context של השרביט תלוי בצבעי הפיקסלים** (`stale_ok=False`); אחרי Commit/Undo/Redo ההצעה נזרקת והקשר מוכן מחדש (כמו ב-GrabCut).
- ערכי הסבילות: ברירת מחדל 20 (כוונן ב-`tools/m4/wand_samples.py`), טווח הסליידר 0..50 (מעל כ-30 מתחילה גלישה בחלק מהחומרים).
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, Signal

from ..select import MagicWandProvider, Point, Prompts
from ..select.wand import DEFAULT_TOLERANCE
from ..session.mask_history import MaskHistory
from .select_tool import SmartSelectController

MAX_UI_TOLERANCE = 50        # טווח הסליידר המומלץ
DRAG_PX_PER_STEP = 4         # פיקסלי מסך לכל יחידת סבילות בגרירה


class WandController(SmartSelectController):
    PROPOSAL_MSG = "select.proposal_wand"
    EMPTY_MSG = "select.empty_wand"

    toleranceChanged = Signal(int)

    def __init__(self, parent=None, *, history_source: Callable[[], MaskHistory | None] | None = None,
                 busy: Callable[[], bool] | None = None, provider_factory: Callable = MagicWandProvider) -> None:
        super().__init__(parent, provider_factory=provider_factory, history_source=history_source, busy=busy)
        self.tolerance = DEFAULT_TOLERANCE
        self._base_tol = self.tolerance

    @property
    def seed(self) -> QPointF | None:
        """נקודת הלחיצה של ההצעה הנוכחית (פיקסלי תמונה רציפים), או None."""
        p = self._prompts
        return QPointF(p.points[0].x, p.points[0].y) if p is not None and p.points else None

    def press(self, pt: QPointF) -> bool:
        """לחיצה בנקודת תמונה רציפה: הצעה חדשה עם הסבילות הנוכחית, והתחלת גרירת סבילות."""
        if self.session is None:
            return False
        self._base_tol = self.tolerance
        return self._submit(Prompts(points=(Point(pt.x(), pt.y()),), tolerance=self.tolerance))

    def drag(self, dx: float) -> None:
        """הסטה אופקית במסך מרגע הלחיצה (ימינה = סבילות גבוהה יותר)."""
        if self.seed is not None:
            self.set_tolerance(self._base_tol + round(dx / DRAG_PX_PER_STEP))

    def set_tolerance(self, value: int) -> None:
        v = min(max(int(value), 0), MAX_UI_TOLERANCE)
        if v == self.tolerance:
            return
        self.tolerance = v
        self.toleranceChanged.emit(v)
        seed = self.seed
        if seed is not None:
            self._submit(Prompts(points=(Point(seed.x(), seed.y()),), tolerance=v))
